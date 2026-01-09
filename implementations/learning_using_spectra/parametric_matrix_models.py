from enum import StrEnum
from typing import Optional

import tensorflow as tf


class EigenSelection(StrEnum):
    smallest_algebraic = "SA"
    largest_algebraic = "LA"
    smallest_magnitude = "SM"
    largest_magnitude = "LM"
    exterior_algebraically = "EA"
    exterior_by_magnitude = "EM"
    interior_algebraically = "IA"
    interior_by_magnitude = "IM"


class SymmetricMatrix(tf.keras.constraints.Constraint):
    # Enforce Hermitian matrix
    def call(self, w):
        return 0.5 * (w + tf.transpose(w))


class AffineHermitianMatrix(tf.keras.layers.Layer):
    def __init__(
        self,
        matrix_size: int,
        num_features: int,
        init_scale: float = 1e-2,
        jitter: float | None = None,
        enforce_hermitian: bool = True,
        bias_term: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.matrix_size = matrix_size
        self.num_features = num_features
        self.init_scale = init_scale
        self.jitter = jitter
        self.enforce_hermitian = enforce_hermitian
        self.bias_term = bias_term

        # number of matrices = features + 1 if bias_term else features
        self.q = self.num_features + 1 if self.bias_term else self.num_features

    def build(self, input_shape):
        self.M = self.add_weight(
            shape=(self.q, self.matrix_size, self.matrix_size),
            trainable=True,
            name="projection_matrices",
            constraint=SymmetricMatrix() if self.enforce_hermitian else None,
            initializer=tf.keras.initializers.RandomNormal(stddev=self.init_scale),
        )

    def call(self, x):
        # Create projection matrix
        if self.bias_term:
            ones = tf.ones([tf.shape(x)[0], 1], dtype=x.dtype)
            x = tf.concat([ones, x], axis=-1)
        else:
            x = x

        # Sum over N square matrices weighted by the features
        # Note: Adding bias is equivalent to adding M0 + sum_i x_i M_i, 1 <= i <= 3
        projection_matrix = tf.einsum("Bi,ijk->Bjk", x, self.M)  # (B, n, n)

        # Optional small diagonal jitter to stabilize eigenspectrum
        if self.jitter and self.jitter > 0:
            eye = tf.eye(self.matrix_size, dtype=x.dtype)[None, :, :]
            projection_matrix += self.jitter * eye

        return projection_matrix


def _select_indices(sorted_idx: tf.Tensor, k: int, which: str) -> tf.Tensor:
    """
    Which eigenvectors to use based on eigenvalue.
    Options are:
    - 'SA' for smallest algebraic
    - 'LA' for largest algebraic
    - 'SM' for smallest magnitude
    - 'LM' for largest magnitude
    - 'EA' for exterior algebraically
    - 'EM' for exterior by magnitude
    - 'IA' for interior algebraically
    - 'IM' for interior by magnitude
    """
    # sorted_idx: (batch, n) indices that sort eigenvalues ascending by chosen metric
    # returns (batch, k) indices

    which = which.upper()
    n = tf.shape(sorted_idx)[-1]

    match which[0].lower():
        case "s":  # Smallest
            return sorted_idx[:, :k]
        case "l":  # Largest
            return sorted_idx[:, -k:]
        case "e":  # Exterior: half of each end
            k_half = k // 2
            k_rem = k - k_half
            left = sorted_idx[:, :k_half]
            right = sorted_idx[:, -k_rem:]
            return tf.concat([left, right], axis=-1)
        case "i":  # Interior: around the centre
            n_half = n // 2
            k_half = k // 2
            k_rem = k - k_half
            left = sorted_idx[:, n_half - k_half : n_half]
            right = sorted_idx[:, n_half : n_half + k_rem]
            return tf.concat([left, right], axis=-1)
        case _:  # Else
            raise ValueError("which must start with one of S, L, E, or I.")


class EigenvaluesSelector(tf.keras.layers.Layer):
    def __init__(
        self,
        num_eig: int | None = 1,  # None returns all
        which: str | EigenSelection = "SA",
        return_vectors: bool = False,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if num_eig is not None and (not isinstance(num_eig, int) or num_eig <= 0):
            raise ValueError("num_eig must be a positive int or None")

        if which not in EigenSelection:
            raise ValueError(
                f"which must be one of {list(EigenSelection._value2member_map_.keys())}"
            )

        self.num_eig = num_eig
        self.which = EigenSelection(which)
        self.return_vectors = return_vectors

    def call(self, M: tf.Tensor) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        # M shape: (batch, n, n)
        if M.shape.rank != 3 or M.shape[-1] != M.shape[-2]:
            raise ValueError("Input must be a batch of square matrices (b, n, n)")

        # Compute eigenvalues and eigenvectors
        E, V = tf.linalg.eigh(M)

        if self.num_eig is None:
            return (E, V) if self.return_vectors else E

        # Choose eigenvalue sort by algebraic (A) or magnitude (M)
        if self.which.value.endswith("A"):
            # algebraic sort
            sorted_idx = tf.argsort(E, axis=-1, stable=True)
        else:
            # magnitude sort
            sorted_idx = tf.argsort(tf.abs(E), axis=-1, stable=True)

        select_idx = _select_indices(sorted_idx, self.num_eig, self.which)

        # Gather selected eigenvalues and vectors per batch
        # First sort columns by sorted_idx, then pick according to idx
        E_sorted = tf.gather(E, sorted_idx, batch_dims=1, axis=-1)  # (b, n)
        E_sel = tf.gather(E_sorted, select_idx, batch_dims=1, axis=-1)  # (b, k)

        if V is None or not self.return_vectors:
            return E_sel

        V_sorted = tf.gather(V, sorted_idx, batch_dims=1, axis=-1)  # (b, n, n)
        V_sel = tf.gather(V_sorted, select_idx, batch_dims=1, axis=-1)  # (b, n, k)

        return E_sel, V_sel


class AffineEigenvaluePMM(tf.keras.layers.Layer):
    def __init__(
        self,
        matrix_size: int,
        num_features: int,
        enforce_hermitian: bool = True,
        bias_term: bool = True,
        init_scale: float = 1e-2,
        jitter: float = 0.0,
        num_eig: int = 1,
        which: str = "SA",
        agg_eigs: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.agg_eigs = agg_eigs

        self.affine = AffineHermitianMatrix(
            matrix_size=matrix_size,
            num_features=num_features,
            enforce_hermitian=enforce_hermitian,
            bias_term=bias_term,
            init_scale=init_scale,
            jitter=jitter,
            name="aff_proj" if name is None else f"aff_proj_{name}",
        )

        self.selector = EigenvaluesSelector(
            num_eig=num_eig,
            which=which,
            return_vectors=False,  # Explicitly false in this learning scenario
            name="eigen_selection" if name is None else f"eigen_selection{name}",
        )

    def call(self, features: tf.Tensor):
        # Project
        M = self.affine(features)

        # Compute eigenvalues
        eigs = self.selector(M)

        # Aggregate
        if self.agg_eigs:
            return tf.keras.ops.sum(eigs, axis=1, keepdims=True)
        else:
            return eigs


if __name__ == "__main__":
    # TODO: Add test code here
    pass

"""Core computational modules used by GB-DivRW.

These functions expose the three central mechanisms of the method: division
profile construction, ball-level dissimilarity propagation, and intra-ball
residual ranking. They are provided as research-oriented building blocks.
"""

from __future__ import annotations

import numpy as np


def division_profile(values: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Return an attribute-wise division profile for an encoded feature matrix.

    Parameters
    ----------
    values:
        A normalized numerical matrix with shape ``(n_objects, n_attributes)``.
    radii:
        One adaptive attribute radius per column.
    """
    values = np.asarray(values, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or radii.shape[0] != values.shape[1]:
        raise ValueError("values and radii must have compatible attribute dimensions.")

    n_objects, n_attributes = values.shape
    profile = np.empty((n_objects, n_attributes), dtype=np.float64)
    for attribute in range(n_attributes):
        difference = np.abs(values[:, [attribute]] - values[:, attribute])
        np.fill_diagonal(difference, -np.inf)
        profile[:, attribute] = np.count_nonzero(difference >= radii[attribute], axis=1)
    return _minmax_columns(profile)


def ball_saliency(
    profile_centres: np.ndarray,
    profile_radii: np.ndarray,
    ball_sizes: np.ndarray,
    restart: float = 0.15,
) -> np.ndarray:
    """Propagate ball-level division dissimilarity by a uniform-restart walk."""
    centres = np.asarray(profile_centres, dtype=np.float64)
    radii = np.asarray(profile_radii, dtype=np.float64).reshape(-1)
    sizes = np.asarray(ball_sizes, dtype=np.float64).reshape(-1)
    if centres.ndim != 2 or len(centres) != len(radii) or len(centres) != len(sizes):
        raise ValueError("Ball centres, radii, and sizes must have compatible shapes.")
    if not 0.0 <= restart <= 1.0:
        raise ValueError("restart must be in [0, 1].")

    centre_distance = np.sum((centres[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    dissimilarity = np.sqrt(np.maximum(centre_distance, 0.0)) + radii[:, None] + radii[None, :]
    np.fill_diagonal(dissimilarity, 0.0)
    stationary = _uniform_restart_stationary(dissimilarity, restart)
    small_ball_weight = 1.0 - np.cbrt(np.clip(sizes / sizes.sum(), 0.0, 1.0))
    return _minmax_1d(stationary * small_ball_weight)


def intra_ball_residual(
    attribute_residual: np.ndarray,
    global_profile_residual: np.ndarray,
    local_profile_residual: np.ndarray,
) -> np.ndarray:
    """Fuse attribute, global-profile, and local-profile residual evidence."""
    profile_residual = 0.5 * _minmax_1d(global_profile_residual) + 0.5 * _minmax_1d(local_profile_residual)
    return 0.5 * _minmax_1d(attribute_residual) + 0.5 * _minmax_1d(profile_residual)


def _uniform_restart_stationary(dissimilarity: np.ndarray, restart: float) -> np.ndarray:
    transition = _row_normalize(dissimilarity)
    n_balls = len(transition)
    distribution = np.full(n_balls, 1.0 / n_balls)
    uniform = distribution.copy()
    for _ in range(1000):
        updated = restart * uniform + (1.0 - restart) * (distribution @ transition)
        if np.linalg.norm(updated - distribution, ord=1) <= 1e-12:
            distribution = updated
            break
        distribution = updated
    return distribution / distribution.sum()


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.maximum(np.asarray(matrix, dtype=np.float64), 0.0)
    row_sum = matrix.sum(axis=1)
    for row in np.flatnonzero(row_sum == 0):
        matrix[row] = 1.0
        matrix[row, row] = 0.0
    row_sum = matrix.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return matrix / row_sum[:, None]


def _minmax_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    lower, upper = values.min(), values.max()
    return np.zeros_like(values) if lower == upper else (values - lower) / (upper - lower)


def _minmax_columns(values: np.ndarray) -> np.ndarray:
    lower = values.min(axis=0)
    upper = values.max(axis=0)
    return np.divide(values - lower, upper - lower, out=np.zeros_like(values), where=(upper - lower) != 0)

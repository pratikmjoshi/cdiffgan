#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A collection of bootstrapping functions.

Currently, it contains:
- bootstrap: percentile bootstrapping (supports pairwise and clustering).
"""
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

NP_TORCH = Union[np.ndarray, torch.Tensor]

@typechecked
def scores_as_matrix(*args: Any, numpy: bool = False) -> List[Optional[NP_TORCH]]:
    """
    Makes sure that the scores are in matrix format.

    args    List of vectors/matrices.
    numpy   Whether to convert tensors to numpy.

    Returns:
        List of (numpy) matrices.
    """
    result = []
    for scores in args:
        # flatten one-element list
        if isinstance(scores, list) and len(scores) == 1:
            scores = scores[0]
        # flatten list of lists
        if isinstance(scores, list) and isinstance(scores[0], list):
            scores = flatten_nested_list(scores)
        # list of tensors to tensor
        if isinstance(scores, list) and isinstance(scores[0], torch.Tensor):
            if numpy:
                for i, score in enumerate(scores):
                    scores[i] = score.detach().cpu()
            scores = torch.cat(scores, dim=0)
        # list of arrays to array
        if isinstance(scores, list) and isinstance(scores[0], np.ndarray):
            scores = np.concatenate(scores, axis=0)
        # list of other to array
        if isinstance(scores, list):
            scores = np.asarray(scores)
        # 1-d array/tensor
        if (isinstance(scores, np.ndarray) and scores.ndim == 1) or (
            isinstance(scores, torch.Tensor) and scores.dim() == 1
        ):
            scores = scores[:, None]
        # detach tensor
        if isinstance(scores, torch.Tensor):
            scores = scores.detach()
            if numpy:
                scores = scores.cpu().numpy()
        result.append(scores)
    return result

@typechecked
def bootstrap(
    y_true: np.ndarray,
    y1_hat: np.ndarray,
    y2_true: Optional[np.ndarray] = None,
    y2_hat: Optional[np.ndarray] = None,
    metric_fun: Callable[..., Dict[str, float]] = None,
    clusters: Optional[np.ndarray] = None,
    ci_interval: float = 0.95,
    resamples: int = 2000,
    y1_hat_kwargs: Dict[str, np.ndarray] = None,
    y2_hat_kwargs: Dict[str, np.ndarray] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Find percentile-based bootstrap intervals.

    All array-like objects have to have the same size along the first dimension.

    y_true  The ground truth.
    y1_hat  The prediction of the ground truth.
    y2_true The optional second ground truth (if the data is paired
            but the second sample has a different ground truth).
    y2_hat  The optional second prediction of the ground truth.
    metric_fun  A function which is given a parts of y_true, y_hat,
                y_hat_kwargs. The function has to return a dictionary of
                scalars. Confidence intervals are calculated for all scalars.
    resamples   How many time to resample.
    clusters    Same shape as y_true, indicates which observations belong
                together (paired clusters). To support a paired metrics, the
                caller has to provide the difference in y1_hat and an
                appropriate metric_fun.
    ci_interval Which confidence interval to use.
    y_hat_kwargs    Dictionary of array-like objects. They are passed to the
                    metric fun. y1_hat and y2_hat have to have the same keyword
                    arguments. Meant to have prediction probabilities or sample
                    weights.

    Returns:
        A dictionary with the following fields:
        '1', '2' (if y2_hat is provided), and '1-2' (if y2_hat is provided).
        All values are dictionaries with the keys of the metric_fun. There
        values are dictionaries with scalars as values. The keys are:
        'lower', 'center', and 'upper'.

    """
    # be reproducible
    rng = np.random.RandomState(1)

    # validate input: convert all inputs to matrix form: n x ?
    y_true, y2_true, y1_hat, y2_hat, clusters = scores_as_matrix(
        y_true, y2_true, y1_hat, y2_hat, clusters, numpy=True
    )
    assert metric_fun is not None
    if y2_true is None:
        y2_true = y_true
    if y1_hat_kwargs is None:
        y1_hat_kwargs = {}
    if y2_hat_kwargs is None:
        y2_hat_kwargs = {}
    for key in y1_hat_kwargs:
        y1_hat_kwargs[key], y2_hat_kwargs[key] = scores_as_matrix(
            y1_hat_kwargs[key], y2_hat_kwargs[key], numpy=True
        )

    # cluster data
    if clusters is not None:
        (
            y1_hat,
            y2_hat,
            y_true,
            y2_true,
            y1_hat_kwargs,
            y2_hat_kwargs,
            metric_fun,
        ) = _clustered_bootstrapping(
            y1_hat,
            y2_hat,
            y_true,
            y2_true,
            y1_hat_kwargs,
            y2_hat_kwargs,
            metric_fun,
            clusters,
        )

    # sampling
    scores1: Dict[str, np.ndarray] = {}
    scores2: Dict[str, np.ndarray] = {}
    for i in range(resamples):
        # resample
        index = rng.choice(y_true.shape[0], size=y_true.shape[0], replace=True)

        # determine scores for y1
        scores1 = _add_metrics(
            scores1, y_true, y1_hat, index, y1_hat_kwargs, resamples, i, metric_fun
        )
        # determine scores for y2
        if y2_hat is None:
            continue
        scores2 = _add_metrics(
            scores2, y2_true, y2_hat, index, y2_hat_kwargs, resamples, i, metric_fun
        )

    # prepare output
    result = {}
    result["1"] = _get_percentiles(scores1, ci_interval)
    if y2_hat is not None:
        result["2"] = _get_percentiles(scores2, ci_interval)
        result["1-2"] = _get_percentiles(
            {key: value - scores2[key] for key, value in scores1.items()}, ci_interval
        )
    return result


@typechecked
def _clustered_bootstrapping(
    y1_hat: np.ndarray,
    y2_hat: Optional[np.ndarray],
    y_true: np.ndarray,
    y2_true: Optional[np.ndarray],
    y1_hat_kwargs: Dict[str, np.ndarray],
    y2_hat_kwargs: Dict[str, np.ndarray],
    metric_fun: Callable[..., Dict[str, float]],
    clusters: Optional[np.ndarray],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Callable[..., Dict[str, float]],
]:
    """
    Calculates metrics for each cluster.
    """
    cluster_values = np.unique(clusters)

    # aggregate clusters
    scores1: Dict[str, np.ndarray] = {}
    scores2: Dict[str, np.ndarray] = {}
    for icluster, cluster in enumerate(cluster_values):
        index = clusters == cluster
        scores1 = _add_metrics(
            scores1,
            y_true,
            y1_hat,
            index,
            y1_hat_kwargs,
            cluster_values.size,
            icluster,
            metric_fun,
        )
        if y2_hat is None:
            continue
        scores2 = _add_metrics(
            scores2,
            y2_true,
            y2_hat,
            index,
            y1_hat_kwargs,
            cluster_values.size,
            icluster,
            metric_fun,
        )

    # redefine metric function and scores
    names = tuple(scores1.keys())
    y1_hat = _flatten_values(scores1)
    y2_hat = _flatten_values(scores2)
    y1_hat_kwargs = {}
    y2_hat_kwargs = {}
    y_true = y1_hat
    metric_fun = partial(_clustered_metric_fun, names=names)

    return y1_hat, y2_hat, y_true, y2_true, y1_hat_kwargs, y2_hat_kwargs, metric_fun


@typechecked
def _flatten_values(scores: Optional[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
    if not scores:
        return None
    return np.concatenate([value.reshape(-1, 1) for value in scores.values()], axis=1)


@typechecked
def _clustered_metric_fun(
    y_true: np.ndarray, y_hat: np.ndarray, names: Tuple[str, ...] = (), **kwargs: Any
) -> Dict[str, float]:
    """Only need to return the mean of the previously calculated metrics."""
    return {name: mean for mean, name in zip(y_hat.mean(axis=0), names)}


@typechecked
def _add_metrics(
    scores: Dict[str, np.ndarray],
    y_true: np.ndarray,
    y_hat: np.ndarray,
    index: np.ndarray,
    kwargs: Dict[str, np.ndarray],
    n_samples: int,
    ith_sample: int,
    metric_fun: Callable[..., Dict[str, float]],
) -> Dict[str, np.ndarray]:
    """Accumulate metrics."""
    index = index.reshape(-1)
    scores_ = metric_fun(
        y_true[index, :],
        y_hat[index, :],
        **{key: value[index, :] for key, value in kwargs.items()},
    )
    for key, value in scores_.items():
        if key not in scores:
            scores[key] = np.full(n_samples, float("NaN"))
        scores[key][ith_sample] = value
    return scores


@typechecked
def _get_percentiles(
    scores: Dict[str, np.ndarray], ci_interval: float
) -> Dict[str, Dict[str, float]]:
    """Calculate PCA confidence intervals."""
    result = {}
    quantiles = [(100 - 100 * ci_interval) / 2, 50, (100 + 100 * ci_interval) / 2]
    for key, value in scores.items():
        percentiles = np.percentile(value, quantiles)
        result[key] = {
            "lower": percentiles[0],
            "center": percentiles[1],
            "upper": percentiles[2],
        }
    return result
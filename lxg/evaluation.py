# implement evaluation measures for attributions
import numpy as np
import torch

from itertools import permutations

from lxg.util import _get_outputs

from scipy.special import comb

import torch
import itertools

import numpy as np
from scipy import stats
from sklearn.utils import resample

from joblib import parallel_backend, Parallel, delayed


def is_representative_sample(values, confidence_level=0.95, bootstrap_iterations=1000, ci_threshold=0.05):
    """
    Determines if the sample mean is likely representative of the population mean.

    Parameters:
    - values (list or numpy array): The sample values (e.g., 1000 values, one for each T_k).
    - confidence_level (float): The desired confidence level for the confidence interval (default: 95%).
    - bootstrap_iterations (int): Number of bootstrap resamples to perform (default: 10,000).
    - ci_threshold (float): Maximum allowable width of the confidence interval as a fraction of the mean (default: 5%).

    Returns:
    - bool: True if the sample mean is likely representative, False otherwise.
    """
    # Calculate the sample mean and standard deviation
    sample_mean = np.mean(values)
    sample_std = np.std(values, ddof=1)
    n = len(values)

    # Calculate the standard error of the mean (SEM)
    sem = sample_std / np.sqrt(n)

    # Calculate the z-score for the given confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the confidence interval
    ci_half_width = z_score * sem
    ci = (sample_mean - ci_half_width, sample_mean + ci_half_width)

    # Check if the confidence interval is narrow enough (relative to the mean)
    if (ci[1] - ci[0]) / sample_mean > ci_threshold:
        return False

    # # Bootstrapping to estimate the stability of the sample mean
    # bootstrap_means = np.zeros(bootstrap_iterations)
    # for i in range(bootstrap_iterations):
    #     bootstrap_sample = resample(values, n_samples=n)
    #     bootstrap_means[i] = np.mean(bootstrap_sample)
    #
    # # Calculate the standard deviation of the bootstrap means
    # bootstrap_std = np.std(bootstrap_means)
    #
    # # Calculate the bootstrap confidence interval
    # bootstrap_ci = np.percentile(bootstrap_means,
    #                              [(1 - confidence_level) / 2 * 100, ((1 + confidence_level) / 2)* 100])
    #
    # # Check if the bootstrap confidence interval is narrow enough
    # if (bootstrap_ci[1] - bootstrap_ci[0]) / sample_mean > ci_threshold:
    #     return False

    # If both checks pass, return True
    return True


def estimate_additional_samples(values, desired_ci_width_fraction=0.05, confidence_level=0.95):
    """
    Estimate the number of additional samples required to achieve a desired confidence interval width.

    Parameters:
    - values (list or numpy array): The current sample values.
    - desired_ci_width_fraction (float): Desired width of the confidence interval as a fraction of the mean.
    - confidence_level (float): The desired confidence level for the confidence interval (default: 95%).

    Returns:
    - int: The estimated number of additional samples required.
    """
    # Calculate the current sample mean and standard deviation
    sample_mean = np.mean(values)
    sample_std = np.std(values, ddof=1)
    current_n = len(values)

    # Calculate the current SEM
    current_sem = sample_std / np.sqrt(current_n)

    # Calculate the current z-score
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    # Calculate the current confidence interval width
    current_ci_width = 2 * z_score * current_sem

    # Calculate the desired confidence interval width
    desired_ci_width = sample_mean * desired_ci_width_fraction

    # Estimate the factor by which the SEM must decrease
    reduction_factor = current_ci_width / desired_ci_width

    # Estimate the required sample size
    required_n = (current_n * reduction_factor ** 2)

    # Calculate the additional samples needed
    additional_samples_needed = max(0, int(np.ceil(required_n - current_n)))

    return additional_samples_needed


def analyze_representativeness(values, desired_ci_width_fraction=0.05, confidence_level=0.95,
                               bootstrap_iterations=1000):
    """
    Combines the representative check and sample size estimation into one function.

    Parameters:
    - values (list or numpy array): The sample values.
    - desired_ci_width_fraction (float): Desired width of the confidence interval as a fraction of the mean.
    - confidence_level (float): The desired confidence level for the confidence interval.
    - bootstrap_iterations (int): Number of bootstrap resamples to perform.

    Returns:
    - tuple: (is_representative, additional_samples_needed)
        - is_representative (bool): True if the sample mean is likely representative.
        - additional_samples_needed (int): Number of additional samples needed if not representative, 0 if representative.
    """
    is_representative = is_representative_sample(
        values,
        confidence_level=confidence_level,
        bootstrap_iterations=bootstrap_iterations,
        ci_threshold=desired_ci_width_fraction
    )

    if is_representative:
        return True, 0
    else:
        additional_samples_needed = estimate_additional_samples(
            values,
            desired_ci_width_fraction=desired_ci_width_fraction,
            confidence_level=confidence_level
        )
        return False, additional_samples_needed


def compute_until_representative(initial_samples, desired_ci_width_fraction=0.05, confidence_level=0.95,
                                 max_iterations=100):
    """
    Continuously compute until a representative sample is achieved.

    Parameters:
    - initial_samples (list or numpy array): The initial sample values.
    - desired_ci_width_fraction (float): Desired width of the confidence interval as a fraction of the mean.
    - confidence_level (float): The desired confidence level for the confidence interval.
    - max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    - final_samples (numpy array): The final sample values when the sample is determined to be representative.
    - iteration (int): The number of iterations performed.
    """
    samples = initial_samples.copy()
    iteration = 0

    while iteration < max_iterations:
        is_rep, additional_samples_needed = analyze_representativeness(
            samples,
            desired_ci_width_fraction=desired_ci_width_fraction,
            confidence_level=confidence_level
        )

        if is_rep:
            print(f"Sample mean is likely representative after {iteration + 1} iterations.")
            return samples, iteration + 1

        # Add more samples if needed
        new_samples = np.random.randn(additional_samples_needed)
        samples = np.concatenate([samples, new_samples])

        iteration += 1

    print(f"Reached maximum iterations ({max_iterations}) without achieving a representative sample.")
    return samples, iteration


def compute_random_perturbation_impact(model, sample, label, baseline, k, inference_fn=None, device=None,
                                       bootstrap_iterations=5, confidence_level=0.95, n_dims_max=9):
    """
    function works on a SINGLE SAMPLE.
    Estimate the impact of random perturbations of k features using either all combinations (for n_dims <= n_dims_max)
    or bootstrapping (for n_dims > n_dims_max) and compute the Area Over the Curve (AOC).

    Parameters:
    - model: The model to evaluate.
    - data: Input data to the model.
    - k: Number of features to perturb (default: 0.5 * data.shape[1], ceiled).
    - inference_fn: Optional alternative function to use for inference.
    - mask: Value used to mask the perturbed features.
    - device: Device to perform computations on (e.g., 'cpu' or 'cuda').
    - bootstrap_iterations: Number of bootstrap samples to draw for larger datasets.
    - confidence_level: The desired confidence level for the confidence interval (default: 95%).
    - n_dims_max: The maximum number of dimensions where all combinations will be computed exactly (default: 9).
        the jump from 8->9 is large
        (n_dims, n_permutations):
        (2, 2), (3, 6), (4, 12), (5, 60), (6, 120), (7, 840), (8, 1680), (9, 15120), (10, 30240), (11, 332640), (12, 665280)]

    Returns:
    - mean_aoc: The mean AOC across all combinations or bootstrap samples.
    - ci_lower: Lower bound of the 95% confidence interval.
    - ci_upper: Upper bound of the 95% confidence interval.
    """

    if inference_fn is None:
        inference_fn = model

    if device is None:
        device = next(model.parameters()).device

    if len(sample.shape) == 1:
        sample = sample.unsqueeze(0)
    _, n_dims = sample.shape

    if k is None or k==0.:
        k = int(np.ceil(0.5 * n_dims))
    elif 0 < k <= 1:
        k = int(np.ceil(k * n_dims))

    def make_dummy_attributions(idxs, shape) -> torch.Tensor:
        _dummy_attributions = torch.zeros(shape) - 1.
        for nc, c in enumerate(idxs):
            for ni, i in enumerate(c):
                _dummy_attributions[nc, i] = k - ni
        return _dummy_attributions

    if n_dims <= n_dims_max:
        # Use all combinations
        all_combinations = np.array(list((itertools.permutations(range(n_dims), k))))
        _samples = sample.expand(len(all_combinations), -1)
        _dummy_attributions = make_dummy_attributions(all_combinations, _samples.shape)
        _targets = torch.tensor([label]*len(all_combinations), dtype=torch.int64)
        _baseline = baseline.expand(len(all_combinations), -1)
        aocs, curves = PGI_le_k(attributions=_dummy_attributions, model=model, targets=_targets, k=k,
                                data=_samples, inference_fn=inference_fn, mask=_baseline,
                                device='cpu', return_curves=True)
        n_permutations_used = len(all_combinations)
        is_representative = 'full'

    else:
        # Use bootstrapping
        aocs = []
        curves = []
        iteration = 0
        combinations_per_iteration = 4096
        _collected_means = []
        while iteration < bootstrap_iterations and len(aocs) < 200_000:
            selected_features = np.stack([np.random.choice(n_dims, k, replace=False)
                                          for _ in range(combinations_per_iteration)])
            _samples = sample.expand(len(selected_features), -1)
            _dummy_attributions = make_dummy_attributions(selected_features, _samples.shape)
            _targets = torch.tensor([label]*len(selected_features), dtype=torch.int64)
            _baseline = baseline.expand(len(selected_features), -1)
            _aocs, _curves = PGI_le_k(attributions=_dummy_attributions, model=model, targets=_targets, k=k,
                                    data=_samples, inference_fn=inference_fn, mask=_baseline,
                                device='cpu', return_curves=True)
            curves.extend(_curves)
            aocs.extend(_aocs)

            agg_aocs = np.sum(np.stack(aocs), axis=1)
            _collected_means.append(np.mean(np.sum(np.stack(_aocs), axis=1)))
            # mean_aoc, ci_lower, ci_upper = calculate_confidence_interval(agg_aocs, confidence_level)
            is_representative, combinations_per_iteration = analyze_representativeness(agg_aocs)

            if is_representative: #abs((ci_upper - ci_lower) / mean_aoc) <= 1-confidence_level:
                break
            # clip number of combinations between 500 and 20k; estimation of samples needed goes down slowly towards the end
            combinations_per_iteration = max(100, min(combinations_per_iteration, 100_000))

            iteration += 1
        n_permutations_used = len(aocs)
    aocs = np.stack(aocs)
    curves = np.stack(curves)
    mean_curves = np.mean(curves, axis=0)

    # Calculate mean AOC and final confidence interval
    mean_aoc, ci_lower, ci_upper = calculate_confidence_interval(aocs, confidence_level)

    # return mean_curves, mean_aoc, ci_lower, ci_upper, n_permtations_used
    return {'mean_aoc': mean_aoc, 'mean_curves': mean_curves,
            'ci_lower': ci_lower, 'ci_upper': ci_upper,
            'n_permutations_used': n_permutations_used, 'is_representative': is_representative}

def calculate_confidence_interval(aocs, confidence_level):
    """
    Calculate the mean AOC and its confidence interval.

    Parameters:
    - aocs: List or array of AOCs.
    - confidence_level: The desired confidence level for the interval.

    Returns:
    - mean_aoc: Mean AOC value.
    - ci_lower: Lower bound of the confidence interval.
    - ci_upper: Upper bound of the confidence interval.
    """
    _aocs = np.sum(aocs, 1)
    mean_aoc = np.mean(_aocs)
    ci_lower, ci_upper = np.percentile(_aocs, [(1 - confidence_level) / 2 * 100, ((1 + confidence_level) / 2)* 100])
    return mean_aoc, ci_lower, ci_upper

def random_baseline_deletion_curves(inference_fn, data, baselines, targets, model, k=0.5, n_jobs=8):
    '''
    given model and data, compute deletion scores for random 'explanations' to obtain a reference value/ curve.

    :param inference_fn:
    :param data:
    :param baselines:
    :param targets:
    :param model:
    :param k:
    :return:
    '''


    # single baseline vector
    if baselines.shape != data.shape:
        baselines = np.repeat(baselines, len(data), 0)

    # results = []
    # for x, y, b in zip(data, targets, baselines):
    #     x = x.unsqueeze(0)
    #     b = b.unsqueeze(0)
    #     y = y.item()
    #     r = compute_random_perturbation_impact(model=model, inference_fn=inference_fn,
    #                                                                    sample=x, label=y, baseline=b, k=k)
    #     results.append(r)

    with parallel_backend(backend='loky', n_jobs=n_jobs):  # 14
        # Parallel()(delayed(rename)(task=a[0], expl_method=a[1], model_set=a[2], gely_threshold=a[3],
        #                                         significance_threshold=a[4], k_means_max_bins=a[5]) for a in arg_sets)
        results = Parallel(verbose=10, batch_size=4)(delayed(compute_random_perturbation_impact)(
            model=model, inference_fn=inference_fn, sample=x, label=y, baseline=b, k=k
        ) for x, y, b in zip(data, targets, baselines))
    import pandas as pd
    results = pd.DataFrame(results)
    return results

def PGI_all_k(model, data, targets, k=2, inference_fn=None, mask=None):
    results = {}
    n_samples, n_dims = data.shape
    n_inferences = 0
    for i in range(1, k+1):
        n_inferences += comb(n_dims, i, exact=True)
    # logging.debug(f"computing PGIOpt for {n_samples} with {n_inferences} each (= {n_samples*n_inferences})")

    device = next(model.parameters()).device
    odds = _get_outputs(inference_fn, data, model, device, batch_size=256)
    if targets is None:
        targets = torch.argmax(odds, dim=1)
    odds = torch.gather(odds, 1, targets.unsqueeze(1)).squeeze()

    for _k in range(1, k+1):
        results[_k] = __PGI_all_k(model, data, odds, targets, k=_k, inference_fn=inference_fn, mask=mask)
    return results


def __PGI_all_k(model, data, odds, targets, k=None, inference_fn=None, mask=None):
    '''
    computes pgi values for all <=k coalitions, saves them with {'coalition':val}

    for each sample in data, compute the coalition of features of size k that maximizes PGI_masked

    computing for k yields partial results for k+1, this should be usable to speed things up;
    might need some hashing/ indexing magic to beat brute force approach that just batches redundant inferences?

    :param model:
    :param data:
    :param targets:
    :param attributions:
    :param k:
    :param inference_fn:
    :param normalize_by_initial_prob:
    :param min_init_prob:
    :param mask:
    :return:
    '''

    if inference_fn is None:
        inference_fn = model

    _prev_model_state = model.training

    model.eval()

    device = next(model.parameters()).device
    n_samples, n_dims = data.shape

    if k is None or k == 0:
        raise ValueError
    elif 0 < k < 1:
        _k = torch.ceil(torch.tensor(k)*n_dims).to(torch.int)
    else:
        _k = k


    _idxs_coalitions = np.array(list(permutations(range(n_dims), _k)))
    n_coalitions = len(_idxs_coalitions)
    print(f"n_coalitions: {n_coalitions}")

    n_steps = _k + 1
    PGIs = []

    with torch.no_grad():
        _data = data.detach().clone()
        for i, (x, y) in enumerate(zip(_data, targets)):
            probs = torch.full(size=(n_coalitions, _k + 1), fill_value=torch.nan)
            probs[:, 0] = odds[i]
            _X = x.repeat((n_coalitions, 1))
            for i in torch.arange(0, n_steps-1):  # go through all k steps
                _idxs_i = _idxs_coalitions[:, i]
                for j in range(n_coalitions):  # mask next feature
                    _X[j, _idxs_i[j]] = mask

                _pred = _get_outputs(inference_fn, _X, model, device, batch_size=256).detach().cpu()
                probs[:, i+1] = _pred[:, y]
            del _X
            assert not probs.isnan().any() and torch.all(probs >= 0) and torch.all(probs <= 1)
            probs = probs.detach().numpy()
            diffs = np.expand_dims(probs[:, 0], 1) - probs[:, 1:]
            _pgis_x = np.sum(diffs, 1)
            PGIs.append(list(zip(_idxs_coalitions, _pgis_x)))

    del _data

    model.train(_prev_model_state)

    return PGIs

def PGI_le_k(model, data, targets, attributions, k=None, inference_fn=None, mask=None, device=None, return_curves=False):
    # compute model output with perturbed input for all steps less or equal to k

    if inference_fn is None:
        inference_fn = model

    _prev_model_state = model.training

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n_samples = data.shape[0]
    n_dims = attributions.shape[1]

    if k is None or k == 0:
        _k = n_dims
    elif 0 < k < 1:
        _k = torch.ceil(torch.tensor(k)*n_dims).to(torch.int)
    else:
        _k = k

    if type(attributions) == np.ndarray:
        attributions = torch.from_numpy(attributions)

    _idxs = torch.argsort(attributions, dim=1, descending=True)[:, :_k]

    n_steps = _idxs.shape[1]+1


    odds = _get_outputs(inference_fn, data, model, device, batch_size=256)
    if targets is None:
        targets = torch.argmax(odds, dim=1)
    odds = torch.gather(odds, 1, targets.unsqueeze(1)).squeeze()

    probs = torch.full(size=(n_samples, n_steps), fill_value=torch.nan)
    probs[:, 0] = odds

    _data = data.detach().clone()
    # _data2 = data.detach().clone()
    batch_indices = torch.arange(_data.size(0), device=device).unsqueeze(1)

    with torch.no_grad():
        for i in torch.arange(0, n_steps-1):
            # # insert mask token
            _idxs_i = _idxs[:, i]
            # for j in range(data.shape[0]):  # how to avoid this loop with sth like data[:, _idxs_i] = masking_value ?
            #     _data[j, _idxs_i[j]] = mask[j, _idxs_i[j]]

            # Expand _idxs to be of shape (n_samples, k, 1) to match dimensions
            expanded_idxs = _idxs_i.unsqueeze(-1)
            # Scatter the mask values into _data at the locations specified by _idxs
            # _data.scatter_(1, expanded_idxs, torch.gather(mask, 1, expanded_idxs))
            # _data2.scatter_(1, expanded_idxs, torch.gather(mask, 1, expanded_idxs))
            _data[batch_indices, expanded_idxs] = mask[batch_indices, expanded_idxs]
            # assert torch.all(_data.eq(_data2))


            # without .detach() memory consumption can balloon (lstm ran oom on 126gb)
            _pred = _get_outputs(inference_fn, _data, model, device, batch_size=256).detach()
            _prob = torch.gather(_pred, 1, targets.unsqueeze(1)).squeeze()
            probs[:, i+1] = _prob

    del _data

    assert not probs.isnan().any(),  f"some probabilities found to be smaller than 0: {probs[torch.argwhere(probs<0)]}"

    probs = probs.detach().numpy()
    AOCs = []
    for i in range(1, n_steps):
        _p_step = probs[:, 1:i+1]
        diffs = np.expand_dims(probs[:, 0], 1) - _p_step
        auc = np.sum(diffs, 1)
        AOCs.append(auc)
    AOCs = np.vstack(AOCs).T  # each row a sample, each column the pgi-k value of idx+1 (col 0 -> k=1)
    model.train(_prev_model_state)
    if return_curves:
        return AOCs, probs

    return {'targets': targets, 'prediction': odds, 'pgi': AOCs}



def PGI(model, data, targets, attributions, k=None, inference_fn=None, n_noise_samples=50, std=1.,
        normalize_by_initial_prob=True, min_init_prob=None, mask=None, device=None):

    '''
    :param model: torch.nn.Module
    :param data: original data
    :param targets: target label for each attribution
    :param attributions: attributions to be evaluated
    :param k: integer or fraction; if 0 or None all dims are used; number of steps and dims covered
    :param inference_fn: callable that takes a batched input
    :param n_noise_samples:
    :param std:
    :param normalize_by_initial_prob:
    :param min_init_prob: minimum certainty of a model for the original sample, samples with smaller certainties will be ignored
    :param mask: single real value used as a masked; if mask is given, n_noise_sampless and std is ignored
    :return:
    '''

    if inference_fn is None:
        inference_fn = model

    if n_noise_samples == 0 or std == 0. or mask is not None:
        n_noise_samples = 1
        std = 0
        if mask is None:
            raise ValueError("n_noise_samples or std was 0 but no mask was provided")

    _prev_model_state = model.training

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    n_samples = data.shape[0]
    n_dims = attributions.shape[1]

    if k is None or k == 0:
        _k = n_dims
    elif 0 < k < 1:
        _k = torch.ceil(torch.tensor(k)*n_dims).to(torch.int)
    else:
        _k = k

    if type(attributions) == np.ndarray:
        attributions = torch.from_numpy(attributions)

    _idxs = torch.argsort(attributions, dim=1, descending=True)[:, :_k]

    n_steps = _idxs.shape[1]+1


    odds = _get_outputs(inference_fn, data, model, device, batch_size=256)
    if targets is None:
        targets = torch.argmax(odds, dim=1)
    odds = torch.gather(odds, 1, targets.unsqueeze(1)).squeeze()
    if min_init_prob is not None:
        _idxs_clears_certainty_threshold = torch.argwhere(odds >= min_init_prob).squeeze()
        _idxs_not_clears_certainty_threshold = torch.argwhere(odds <= min_init_prob).squeeze()
        assert len(_idxs_clears_certainty_threshold) > 0
        n_samples_new = len(_idxs_clears_certainty_threshold)
        data = data[_idxs_clears_certainty_threshold]
        targets = targets[_idxs_clears_certainty_threshold]
        odds = odds[_idxs_clears_certainty_threshold]
        # print(f"pgi removed {n_samples - n_samples_new} of {n_samples} because of certainty threshold")
        n_samples = n_samples_new
    probs = torch.full(size=(n_samples, n_steps), fill_value=torch.nan)
    probs[:, 0] = odds

    for _ in range(n_noise_samples):
        _data = data.detach().clone()
        with torch.no_grad():
            for i in torch.arange(0, n_steps-1):
                # insert mask token
                _idxs_i = _idxs[:, i]
                for j in range(data.shape[0]):  # how to avoid this loop with sth like data[:, _idxs_i] = masking_value ?
                    if mask is not None:
                        _data[j, _idxs_i[j]] = mask
                    else:
                        _data[j, _idxs_i[j]] += np.random.normal(0, std)
                # without .detach() memory consumption can balloon (lstm ran oom on 126gb)
                _pred = _get_outputs(inference_fn, _data, model, device, batch_size=256).detach()
                _prob = torch.gather(_pred, 1, targets.unsqueeze(1)).squeeze()
                assert torch.all(_prob <= 1.)
                if probs[:, i+1].isnan().all():  # == doesnt work and 'is' is not applied element wise
                    probs[:, i+1] = _prob
                else:
                    probs[:, i+1] += _prob

    del _data

    assert not probs.isnan().any(),  f"some probabilities found to be smaller than 0: {probs[torch.argwhere(probs<0)]}"

    if normalize_by_initial_prob:
        probs = probs * 1/probs[:, 0].unsqueeze(1)
    probs[:, 1:] /= n_noise_samples

    probs = probs.detach().numpy()
    diffs = np.expand_dims(probs[:, 0], 1) - probs[:, 1:]
    auc = np.sum(diffs, 1)

    if min_init_prob is not None:
        auc = [a for a in auc]
        for idx in _idxs_not_clears_certainty_threshold:
            auc.insert(idx, np.NaN)
        auc = np.array(auc)
        assert np.all(np.argwhere(auc == np.NaN).squeeze() == _idxs_not_clears_certainty_threshold)

    model.train(_prev_model_state)

    return auc

# make computing pgi for several k more efficient by saving partial results for k-1, k-2 ...
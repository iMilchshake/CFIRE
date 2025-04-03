from itertools import combinations
import torch
import numpy as np

# --- HELPER FUNCTIONS

def __greedy_pgi_attribution(sample, target, init_prob, pgi_fn, min_k=1, max_k=None):
    n_dims = len(sample)

    if max_k == None:
        max_k = int(n_dims * 0.33)

    _dims = set(np.arange(n_dims))

    _init_combinations = np.array([c for c in combinations(list(_dims), min_k)])
    a = np.zeros((len(_init_combinations), n_dims))

    for i, co in enumerate(_init_combinations):
        for c in co:
            a[i, c] = 1

    gaps = [pgi_fn(sample, _a, target, init_prob) for _a in a]
    best_score = np.max(gaps)
    top_a = a[np.argmax(gaps)]
    if min_k == 1:
        top_k = [np.argwhere(top_a == 1).squeeze().item()]
    else:
        top_k = list(np.argwhere(top_a == 1).squeeze())

    for _ in range(min_k+1, max_k):
        _dims = _dims - set(top_k)
        a = np.zeros((len(_dims), n_dims))
        a[:, top_k] = 1
        for i, d in enumerate(_dims):
            a[i, d] = 1

        gaps = [pgi_fn(sample, _a, target, init_prob) for _a in a]
        _top_score = np.max(gaps)
        if best_score < _top_score:
            best_score = _top_score
            top_a = a[np.argmax(gaps)]
            _top_k = np.argwhere(top_a == 1).squeeze()
            for _k in _top_k:
                if _k not in top_k:
                    top_k.append(_k)
        else:
            break

    # improvise a weighting
    a = np.zeros(n_dims)
    for i, k in enumerate(top_k):
        if i < min_k:
            a[k] = len(top_k)
        else:
            a[k] = len(top_k) - i
    a = a/len(top_k)

    return a

def _default_pgi_fn(model, sample, target, init_prediction, binary_mask, masking_value=0.,
                   inference_fn=None):

    # binary_mask: 1 == cover feature
    idxs = np.where(binary_mask == 1)
    sample = sample.detach().clone()
    sample[idxs] = masking_value

    if inference_fn is None:
        inference_fn = model

    _prev_model_state = model.training
    model.eval()

    if len(sample.shape) < 2:
        sample = torch.unsqueeze(sample, 0)
    p = inference_fn(sample)[0, target].detach().numpy().item()
    gap = init_prediction - p

    model.train(_prev_model_state)

    return gap

# ---

def greedy_pgi_attribution(model, data, targets=None, inference_fn=None, device=None, pgi_fn=None):

    if inference_fn is None:
        inference_fn = model

    if pgi_fn is None:
        pgi_fn = lambda x, m, t, p: _default_pgi_fn(
            model=model, sample=x, target=t, init_prediction=p, binary_mask=m, inference_fn=inference_fn
        )

    if targets is None:
        init_probs = inference_fn(data)
        targets = torch.argmax(init_probs, dim=1)
        init_probs = init_probs[:, targets].detach().numpy()
    else:
        init_probs = torch.gather(inference_fn(data), 1, targets.unsqueeze(1)).squeeze().detach().numpy()

    attributions = []
    for x, y, ip in zip(data, targets, init_probs):
        attributions.append(
            __greedy_pgi_attribution(
                sample=x, init_prob=ip, target=y, pgi_fn=pgi_fn, min_k=1, max_k=None
            )
        )
    attributions = torch.Tensor(attributions)
    return attributions
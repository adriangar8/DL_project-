import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import prunable_named_parameters

def snip_mask(model: nn.Module, data: torch.Tensor, target: torch.Tensor, sparsity: float):
    """
    SNIP score: |w * dL/dw|, global top-k keep.
    sparsity=0.9 -> 90% prune
    """
    model.zero_grad(set_to_none=True)
    out = model(data)
    loss = F.cross_entropy(out, target)
    loss.backward()

    flat_scores = []
    per_param_scores = {}

    for name, p in prunable_named_parameters(model):
        if p.grad is None:
            continue
        score = (p.grad * p).abs().detach()
        per_param_scores[name] = score
        flat_scores.append(score.reshape(-1))

    all_scores = torch.cat(flat_scores)
    total = all_scores.numel()
    k_keep = max(1, int(total * (1.0 - sparsity)))

    kth = total - k_keep + 1
    threshold, _ = torch.kthvalue(all_scores, kth)

    mask_dict = {}
    for name, score in per_param_scores.items():
        mask_dict[name] = (score >= threshold).to(dtype=score.dtype)

    return mask_dict
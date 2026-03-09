import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import prunable_named_parameters

def grasp_mask(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    sparsity: float,
    use_abs: bool = True,
):
    """
    GraSP (stabil, repo-kompatibilis):
    g = dL/dw (create_graph=True)
    hvp ~ d/dw (||g||^2)  (rankinghez elég)
    score ~ -w * hvp
    global top-k keep -> mask

    use_abs=True: stabilabb gyakorlatban.
    """

    named = list(prunable_named_parameters(model))
    names = [n for n, _ in named]
    params = [p for _, p in named]

    model.zero_grad(set_to_none=True)
    out = model(data)
    loss = F.cross_entropy(out, target)

    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    grad_norm = torch.zeros((), device=data.device)
    for g in grads:
        grad_norm = grad_norm + (g * g).sum()

    hvp = torch.autograd.grad(grad_norm, params, create_graph=False, retain_graph=False)

    scores_per_param = {}
    flat_scores = []
    for name, p, h in zip(names, params, hvp):
        s = (-p * h).detach()
        if use_abs:
            s = s.abs()
        scores_per_param[name] = s
        flat_scores.append(s.reshape(-1))

    all_scores = torch.cat(flat_scores)
    total = all_scores.numel()
    k_keep = max(1, int(total * (1.0 - sparsity)))

    kth = total - k_keep + 1
    threshold, _ = torch.kthvalue(all_scores, kth)

    mask_dict = {}
    for name, s in scores_per_param.items():
        mask_dict[name] = (s >= threshold).to(dtype=s.dtype)

    return mask_dict
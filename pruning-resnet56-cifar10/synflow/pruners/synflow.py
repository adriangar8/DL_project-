import torch
import torch.nn as nn

from .utils import prunable_named_parameters

def _linearize_(model: nn.Module):
    """
    SynFlow trükk: abs súlyokkal dolgozunk (pozitív háló),
    és eltesszük a sign-okat visszaállításhoz.
    """
    signs = {}
    with torch.no_grad():
        for name, p in prunable_named_parameters(model):
            signs[name] = torch.sign(p)
            p.abs_()
    return signs

def _restore_(model: nn.Module, signs: dict):
    with torch.no_grad():
        for name, p in prunable_named_parameters(model):
            p.mul_(signs[name])

def synflow_scores(model: nn.Module, input_shape=(1, 3, 32, 32), device=None):
    """
    SynFlow score: |w * dR/dw| ahol R = sum(output)
    és input = all-ones.
    """
    if device is None:
        device = next(model.parameters()).device

    signs = _linearize_(model)

    model.zero_grad(set_to_none=True)
    x = torch.ones(input_shape, device=device)

    out = model(x)
    # CIFAR model out shape: [B, num_classes]
    R = out.sum()
    R.backward()

    score_dict = {}
    with torch.no_grad():
        for name, p in prunable_named_parameters(model):
            score_dict[name] = (p.grad * p).abs().detach()

    _restore_(model, signs)
    model.zero_grad(set_to_none=True)

    return score_dict
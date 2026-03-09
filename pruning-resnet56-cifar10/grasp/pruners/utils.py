import torch
import torch.nn as nn

def prunable_named_parameters(model: nn.Module):
    """
    Unstructured SNIP: Conv/Linear weight paraméterek.
    (BN, bias, stb. kihagyva.)
    """
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".weight") and (p.dim() == 4 or p.dim() == 2):
            yield name, p

@torch.no_grad()
def apply_mask_(model: nn.Module, mask_dict: dict):
    for name, p in model.named_parameters():
        if name in mask_dict:
            p.mul_(mask_dict[name].to(p.device))

@torch.no_grad()
def enforce_mask_after_step_(model: nn.Module, mask_dict: dict):
    for name, p in model.named_parameters():
        if name in mask_dict:
            p.mul_(mask_dict[name].to(p.device))
import torch
from . import networks

def backbone_map(backbone):
    if 'dual_clip' in backbone:
        return 'DualStreamCLIP'  # Maps to the new class in networks.py
    elif 'resnet' in backbone:
        return 'resnet'
    elif 'clip' in backbone:
        return 'clip_encoder'
    elif 'safas' in backbone:
        return 'resnet18'
    else:
        # Fallback or error raising could go here
        return 'clip_encoder'

def build_model(args):
    # This automatically calls networks.DualStreamCLIP(args) if backbone is 'dual_clip'
    return getattr(networks, backbone_map(args.backbone))(args)

def build_optimizer(args, net):
    # Use Adam for CLIP-based models (Single or Dual stream)
    if 'clip' in args.backbone or 'dual' in args.backbone:
        return torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        return torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
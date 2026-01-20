from .casia_dataset import CasiaSurfDataset, CefaAFDataset
from .facedataset import FaceDataset, BalanceFaceDataset
from torch.utils.data import DataLoader, ConcatDataset # <--- Added ConcatDataset
from torchvision import transforms
import os

def protocol_decoder(protocol):
    MAP = {
        # Original datasets
        'C': 'CASIA',
        'I': 'Idiap',
        'M': 'MSU',
        'O': 'OULU',
        'A': 'CelebA',
        'W': 'SiW',
        's': 'Surf',
        'c': 'CeFA',
        'w': 'WMCA',
        # Kaggle datasets
        'K': '30k_fas',
        'R': '98k_real',
        'V': 'archive',
        'U': 'unidata_real',
        # Custom datasets
        'S': 'CelebA_Spoof-mini',
        'F': 'FAS_processed',
        'Custom': 'CustomFAS',
    }

    train_protocols, test_protocols = protocol.split('_to_')
    train_protocols = train_protocols.split('_')
    test_protocols = test_protocols.split('_')
    return [MAP[train_protocol] for train_protocol in train_protocols], [MAP[test_protocol] for test_protocol in                                   test_protocols]


def get_transform(backbone):
    if 'resnet' in backbone:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0), ratio=(1., 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.9, 0.9), ratio=(1., 1.)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1., 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.9, 0.9), ratio=(1., 1.)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return train_transform, test_transform


def build_datasets(args):
    train_transform, test_transform = get_transform(args.backbone)

    # --- NEW LOGIC FOR CASIA (DUAL STREAM + CEFA) ---
    if "CASIA" in args.protocol:
        print(f"Loading CASIA-SURF + CeFA Dataset (Dual Stream)...")
        
        # 1. Load CASIA-SURF (From command line arg path)
        surf_train = CasiaSurfDataset(args.data_root, phase='train', transform=train_transform)
        
        # 2. Load CeFA-AF
        # ⚠️ IMPORTANT: Update this string to match your actual local path to CeFA-Race
        cefa_root = os.path.join(os.path.dirname(args.data_root), "CeFA", "CeFA-Race")
        # Alternative: cefa_root = "dataset/CeFA/CeFA-Race"
        
        if os.path.exists(cefa_root):
            cefa_train = CefaAFDataset(cefa_root, phase='train', transform=train_transform)
            # Combine
            combined_train = ConcatDataset([surf_train, cefa_train])
            print(f"Merged: {len(surf_train)} SURF + {len(cefa_train)} CeFA = {len(combined_train)} Total")
        else:
            print(f"⚠️ Warning: CeFA path not found at {cefa_root}. Using only CASIA-SURF.")
            combined_train = surf_train

        # CASIA doesn't support domain generalization logic yet, set num_domain to 1
        args.num_domain = 1 
        
        test_dataset = CasiaSurfDataset(args.data_root, phase='test', transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        
        return train_loader, test_loader
    # -----------------------------------------

    # --- ORIGINAL LOGIC (SINGLE STREAM) ---
    train_protocol_names, test_protocol_names = protocol_decoder(args.protocol)
    args.num_domain = len(train_protocol_names)

    train_dataset = BalanceFaceDataset(args.data_root, train_protocol_names, 'train', train_transform,
                                       args.max_iter * args.batch_size, not args.silence)
    test_dataset = FaceDataset(args.data_root, test_protocol_names, 'test', test_transform, not args.silence)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader
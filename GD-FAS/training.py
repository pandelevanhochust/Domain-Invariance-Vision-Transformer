import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
from data import build_datasets
from models import build_model, build_optimizer
from utils import *


def log_f(f, console=True):
    def log(msg):
        with open(f, 'a') as file:
            file.write(msg)
            file.write('\n')
        if console:
            print(msg)

    return log


def parse_args():
    parser = argparse.ArgumentParser()
    # Key Arguments
    parser.add_argument('--backbone', type=str, default="clip", help='backbone - resnet18, clip, or dual_clip')
    parser.add_argument('--silence', action='store_true')
    parser.add_argument('--log_name', type=str, default="test", help='log directory name')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--data_root', type=str, default="dataset", help='Path to dataset root')
    parser.add_argument('--protocol', type=str, default="Custom_to_Custom", help='S_to_S, Custom_to_Custom, or CASIA')
    parser.add_argument('--max_iter', type=int, default=4000, help='Total training iterations')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.000003, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.000001, help='Weight decay')
    parser.add_argument('--gs', action='store_true', help='Use Group Scaling')
    parser.add_argument('--save', action='store_true', default=True, help='Save checkpoints')
    parser.add_argument('--beta', type=float, default=1.5, help='Hyperparameter beta')
    parser.add_argument('--temperature', type=float, default=0.1, help='Hyperparameter temperature')
    parser.add_argument('--params', nargs=4, type=float, default=[1.0, 0.8, 0.1, 1.0], help='Loss weights')
    parser.add_argument('--step_size', type=int, default=500, help='Step size for LR scheduler')
    parser.add_argument('--num_classes', type=int, default=2, help='2 for Binary (Live/Spoof), 3 for Multi-class')
    parser.add_argument('--num_domain', type=int, default=1, help='Number of domains')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval(list_scores):
    # Basic Evaluation Helper
    preds = []
    labels = []
    for line in list_scores:
        s, l = line.strip().split()
        preds.append(float(s))
        labels.append(int(l))

    preds = np.array(preds)
    labels = np.array(labels)

    # Calculate Accuracy (Threshold 0.5)
    threshold = 0.5
    pred_labels = (preds > threshold).astype(int)
    acc = (pred_labels == labels).mean()

    # Return ACC and dummy placeholders
    return [acc], 0.0, [0.0], 0.0, 0.5, 0.0, acc, 0, 0


def main(args):
    # Logging Setup
    log_path = 'results/{}'.format(args.log_name)
    os.makedirs(log_path, exist_ok=True)
    print_log = log_f(os.path.join(log_path, '{}.txt'.format(args.protocol)))

    # Setup Components
    train_loader, test_loader = build_datasets(args)
    networks = build_model(args)
    optimizer = build_optimizer(args, networks)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    networks.to(device)
    print_log(f'Using device: {device}')

    best_acc = 0.0
    current_iter = 0

    print_log(f"Starting training for {args.max_iter} iterations...")

    # --- INFINITE LOOP LOGIC ---
    while current_iter < args.max_iter:
        for batch_samples in train_loader:
            if current_iter >= args.max_iter:
                break

            networks.train()
            optimizer.zero_grad()

            # 1. DUAL STREAM CASE (Model A)
            # We ONLY enter here if the user specifically requested 'dual_clip'
            if args.backbone == 'dual_clip' and isinstance(batch_samples, dict) and 'image_ir' in batch_samples:
                images_rgb = batch_samples['image_x'].to(device)
                images_ir = batch_samples['image_ir'].to(device)
                labels = batch_samples['label'].to(device)
                domains = torch.zeros_like(labels).to(device)

                # Pass 4 Arguments: (RGB, IR, Label, Domain)
                loss = networks.compute_loss(images_rgb, images_ir, labels, domains)

            # 2. SINGLE STREAM CASE (Model B / IR Only)
            else:
                # Handle dictionary unpacking for single stream
                if isinstance(batch_samples, dict):
                    # For CASIA_IR, the data is in 'image_x'
                    images = batch_samples['image_x'].to(device)
                    labels = batch_samples['label'].to(device)
                    domains = torch.zeros_like(labels).to(device)

                # Handle Legacy BalanceFaceDataset (Nested Dicts)
                elif 'sample' in batch_samples:
                    # ... (Legacy logic for BalanceFaceDataset) ...
                    pass

                    # Pass 3 Arguments: (Image, Label, Domain)
                # This matches 'clip_encoder.compute_loss' signature
                loss = networks.compute_loss(images, labels, domains)

            loss.backward()
            optimizer.step()

            current_iter += 1

            # EVALUATION LOOP
            if (current_iter % 200 == 0):
                scheduler.step()

                try:
                    infos = networks.loss_reset()
                    print_log(f'Iter {current_iter}: {infos["loss"]}')
                except:
                    print_log(f'Iter {current_iter}: Loss {loss.item():.4f}')

                # Run Validation
                networks.eval()
                list_scores = []
                with torch.no_grad():
                    for test_batch in test_loader:
                        # Correct Evaluation Logic
                        # If we are in Dual Mode, use Dual inputs
                        if args.backbone == 'dual_clip' and 'image_ir' in test_batch:
                            img_rgb = test_batch['image_x'].to(device)
                            img_ir = test_batch['image_ir'].to(device)
                            logits = networks(img_rgb, img_ir)
                            lbl = test_batch['label'].to(device)

                        # Else use Single Input
                        else:
                            img = test_batch['image_x'].to(device)
                            lbl = test_batch['label'].to(device)
                            logits, _ = networks(img)

                        probs = torch.nn.functional.softmax(logits, dim=1)
                        for prob, label in zip(probs, lbl):
                            list_scores.append(f"{prob[1].item()} {label.item()}")

                metrics = eval(list_scores)
                acc = metrics[0][0]
                print_log(f"Test ACC: {acc:.4f}")

                if acc >= best_acc:
                    best_acc = acc
                    if args.save:
                        save_name = f'results/{args.log_name}/{args.protocol}_best.pth'
                        torch.save(networks.state_dict(), save_name)
                        print_log(f"Saved best model to {save_name}")

                print_log('------------------------------------------------------')


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)
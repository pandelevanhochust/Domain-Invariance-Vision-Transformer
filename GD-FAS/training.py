import argparse
import os
import random

import numpy
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
    # build dirs
    parser.add_argument('--backbone', type=str, default="clip", help='backbone - resnet18, clip, or dual_clip')
    parser.add_argument('--silence', action='store_true')
    parser.add_argument('--log_name', type=str, default="test", help='log')
    parser.add_argument('--seed', type=int, default=2025, help='')
    parser.add_argument('--data_root', type=str, default="dataset", help='YOUR_Data_Dir')
    parser.add_argument('--protocol', type=str, default="Custom_to_Custom",
                        help='S_to_S for CelebA_Spoof-mini, or Custom_to_Custom')
    parser.add_argument('--max_iter', type=int, default=500, help='')
    parser.add_argument('--batch_size', type=int, default=8, help='')
    parser.add_argument('--lr', type=float, default=0.000003, help='')
    parser.add_argument('--wd', type=float, default=0.000001, help='')
    parser.add_argument('--gs', action='store_true')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--beta', type=float, default=1.5, help='')
    parser.add_argument('--temperature', type=float, default=0.1, help='')
    parser.add_argument('--params', nargs=4, type=float, default=[1.0, 0.8, 0.1, 1.0])
    parser.add_argument('--step_size', type=int, default=10, help='')
    
    # ADDED: Support for multi-class (2 or 3)
    parser.add_argument('--num_classes', type=int, default=2, help='2 for binary, 3 for multi-class')
    
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # print to txt file
    log_path = 'results/{}'.format(args.log_name)
    os.makedirs(log_path, exist_ok=True)
    print = log_f(os.path.join(log_path, '{}.txt'.format(args.protocol)))

    # setup
    train_loader, test_loader = build_datasets(args)
    networks = build_model(args)
    optimizer = build_optimizer(args, networks)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    networks.to(device)
    print(f'Using device: {device}')

    # training
    if not args.silence:
        print('------------------------------------------------------')
        print('information')
        print('------------------------------------------------------')
        print(f'{"log name":20} : {args.log_name}')
        print(f'{"protocol name":20} : {args.protocol}')
        print(f'{"backbone":20} : {args.backbone}')
        print(f'{"batch size":20} : {args.batch_size}')
        print(f'{"learning rate":20} : {args.lr}')
        print(f'{"weight decay":20} : {args.wd}')
        print(f'{"seed":20} : {args.seed}')
        print(f'{"max iter":20} : {args.max_iter}')
        print(f'{"step size":20} : {args.step_size}')
        print(f'{"gs":20} : {args.gs}')
        print(f'{"beta":20} : {args.beta}')
        print(f'{"temperature":20} : {args.temperature}')
        print(f'{"num classes":20} : {args.num_classes}')
        print(f'{"parameter1":20} : {args.params[0]}')
        print(f'{"parameter2":20} : {args.params[1]}')
        print(f'{"parameter3":20} : {args.params[2]}')
        print(f'{"parameter4":20} : {args.params[3]}')
        print('------------------------------------------------------')
        print('training')
        print('------------------------------------------------------')

    best_select = [1, ""]
    
    for iter, batch_samples in enumerate(train_loader):
        # Optional: Uncomment if you want to see progress every batch
        # print(f"Processing batch {iter}...")
        
        epoch = iter // 10
        networks.train()
        optimizer.zero_grad()
        device = next(networks.parameters()).device

        # --- LOGIC SPLIT: DUAL STREAM (CASIA) vs SINGLE STREAM ---
        if isinstance(batch_samples, dict) and 'image_ir' in batch_samples:
            # === CASE 1: Dual Stream (RGB + IR) ===
            image_rgb = batch_samples['image_x'].to(device)
            image_ir = batch_samples['image_ir'].to(device)
            labels = batch_samples['label'].to(device)
            
            # Create dummy domains (CASIA doesn't strictly use domain labels here)
            domains = torch.zeros_like(labels).to(device)
            
            # Compute Loss (Requires 4 args for DualStream)
            loss = networks.compute_loss(image_rgb, image_ir, labels, domains)

        else:
            # === CASE 2: Single Stream (Original) ===
            # This handles 'BalanceFaceDataset' which returns nested dicts
            image_x_v1 = torch.cat([batch_samples[key]['image_x_v1'] for key in batch_samples])
            image_x_v2 = torch.cat([batch_samples[key]['image_x_v2'] for key in batch_samples])

            images = torch.cat([image_x_v1, image_x_v2]).to(device)
            labels = torch.cat([batch_samples[key]['label'] for key in batch_samples]).repeat(2).to(device)
            domains = torch.cat([batch_samples[key]['domain'] for key in batch_samples]).repeat(2).to(device)

            # Compute Loss (Standard 3 args)
            loss = networks.compute_loss(images, labels, domains)

        # ---------------------------------------------------------

        loss.backward()
        optimizer.step()

        if (iter % 10 == 0) & (iter != 0):
            scheduler.step()
            infos = networks.loss_reset()
            print(f'epoch : {epoch} {infos["loss"]}')
            print('------------------------------------------------------------------------------------------------------------')
            
            # --- EVALUATION LOOP ---
            list_scores = []
            networks.eval()
            with torch.no_grad():
                for test_batch in test_loader:
                    device = next(networks.parameters()).device
                    lbl = test_batch['label'].to(device)

                    # Check for Dual Stream Data
                    if 'image_ir' in test_batch:
                         img_rgb = test_batch['image_x'].to(device)
                         img_ir = test_batch['image_ir'].to(device)
                         logits = networks(img_rgb, img_ir)
                    else:
                         images = test_batch['image_x'].to(device)
                         # Returns tuple (logits, classifier_out) or just logits depending on model
                         outputs = networks(images)
                         if isinstance(outputs, tuple):
                             logits = outputs[0]
                         else:
                             logits = outputs

                    # Calculate Probabilities
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    
                    for prob, label in zip(probs, lbl):
                        # SCORE CALCULATION:
                        # prob[0] is Live, prob[1] is Attack (in binary)
                        # In multi-class, prob[0]=Live, prob[1]=Cutout, prob[2]=Replay
                        # Universal Spoof Score = 1.0 - Prob(Live)
                        spoof_score = 1.0 - prob[0].item()
                        
                        # Binary Label for HTER (0=Live, 1=Spoof)
                        binary_label = 1 if label.item() > 0 else 0
                        
                        list_scores.append("{} {}\n".format(spoof_score, binary_label))

                test_ACC, tpr_filtered_1p, HTER, auc_test, val_threshold, val_ece, val_acc, sc, la = eval(list_scores)
                print(
                    "ACC_val:{:.4f} HTER_val:{:.4f} AUC:{:.4f} fpr1p:{:.4f} ECE:{:.4f} acc:{:.4f} threshold:{:.4f} ".format(
                        test_ACC[0], HTER[0], auc_test, tpr_filtered_1p, val_ece, val_acc, val_threshold))

                if best_select[0] >= HTER[0]:
                    best_select[0] = HTER[0]
                    best_select[
                        1] = "ACC_val:{:.4f} HTER_val:{:.4f} AUC:{:.4f} fpr1p:{:.4f} ECE:{:.4f} acc:{:.4f} threshold:{:.4f} ".format(
                        test_ACC[0], HTER[0], auc_test, tpr_filtered_1p, val_ece, val_acc, val_threshold)
                    if args.save:
                        torch.save(networks, f'results/{args.log_name}/{args.protocol}_best.pth')

                print(f'best_hter: {best_select[0]:.4f}')
                print('------------------------------------------------------------------------------------------------------------')
    print(best_select[1])


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)
import os
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*unhashable type: 'list'.*",
    module="torch.onnx._internal._beartype"
)
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
from util import psnr, crop_size
from 图像.WCL.model.architecture import PRAF_JSCC
from 图像.WCL.model.dataset import get_fixed_snr, get_snr_range


def train_praf_jscc(args, model, device):
    train_dataset, val_dataset = get_snr_range(
        args.train_hr_dir, args.train_lr_dir, args.val_hr_dir, args.val_lr_dir,
        snr_low=args.snr_low_train,
        snr_high=args.snr_up_train,
        patch_size=args.patch_size
    )
    if args.pretrained_model and os.path.isfile(args.pretrained_model):
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        print(f"✅ Loaded pretrained model from {args.pretrained_model}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    min_val_loss = float('inf')
    loss_log = {'epoch': [], 'train_loss': [], 'val_loss': []}

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.loss_dir, exist_ok=True)

    best_path = os.path.join(
        args.model_dir,
        f"best_praf_jscc_snr{args.snr_low_train}to{args.snr_up_train}_{args.channel_type}.pth"
    )
    last_path = os.path.join(
        args.model_dir,
        f"last_praf_jscc_snr{args.snr_low_train}to{args.snr_up_train}_{args.channel_type}.pth"
    )

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for lr_img, hr_img, snr in tqdm(train_loader, desc=f"[Epoch {epoch+1}/{args.epochs}] Training"):
            lr_img, hr_img, snr = lr_img.to(device), hr_img.to(device), snr.to(device)

            recon = model(lr_img, snr)
            # recon = recon.clamp(0, 1)
            loss = criterion(recon, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        val_loss = validate_praf_jscc(model, val_loader, criterion, device)
        torch.save(model.state_dict(), last_path)
        # === log ===
        print(f"Epoch {epoch+1:03d} | train_loss={total_train_loss:.6f} | val_loss={val_loss:.6f}")
        loss_log['epoch'].append(epoch + 1)
        loss_log['train_loss'].append(total_train_loss)
        loss_log['val_loss'].append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best model at epoch {epoch+1}")

        with open(os.path.join(args.loss_dir, "train_log.json"), 'w') as f:
            json.dump(loss_log, f)



def validate_praf_jscc(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for lr_img, hr_img, snr in tqdm(loader, desc="Validating"):
            lr_img, hr_img, snr = lr_img.to(device), hr_img.to(device), snr.to(device)
            recon = model(lr_img, snr)
            loss = criterion(recon, hr_img)
            total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_praf_jscc(args, model, device):
    model.eval()
    criterion = nn.MSELoss()

    snr_list, psnr_list, ssim_list = [], [], []

    for snr in range(args.snr_low_eval, args.snr_up_eval + 1, 2):
        print(f"\n=== Evaluating at {snr} dB ===")

        test_dataset, _ = get_fixed_snr(
            hr_dir_val=args.val_hr_dir,
            lr_dir_val=args.val_lr_dir,
            snr=snr,
            patch_size=None
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        total_mse, total_ssim = 0.0, 0.0

        with torch.no_grad():
            for lr_img, hr_img, snr_tensor in tqdm(test_loader, desc=f"Testing {snr} dB"):
                lr_img, hr_img, snr_tensor = lr_img.to(device), hr_img.to(device), snr_tensor.to(device)

                recon = model(lr_img, snr_tensor)
                recon, hr_img = crop_size(recon, hr_img)
                mse_loss = criterion(recon, hr_img)
                total_mse += mse_loss.item()

                ssim_val = ssim(recon, hr_img, data_range=1.0, size_average=True)
                total_ssim += ssim_val.item()

        mean_mse = total_mse / len(test_loader)
        mean_ssim = total_ssim / len(test_loader)
        psnr_val = psnr(mean_mse)

        snr_list.append(snr)
        psnr_list.append(psnr_val)
        ssim_list.append(mean_ssim)

        print(f"SNR {snr:2d} dB | PSNR {psnr_val:.6f} dB | SSIM {mean_ssim:.6f}")

    os.makedirs(args.eval_dir, exist_ok=True)
    df = pd.DataFrame({
        'SNR(dB)': snr_list,
        'PSNR(dB)': psnr_list,
        'SSIM': ssim_list
    })
    csv_filename = f"praf_jscc_eval_snr{args.snr_low_eval}to{args.snr_up_eval}_{args.channel_type}.csv"
    df.to_csv(os.path.join(args.eval_dir, csv_filename), index=False)
    print(f"\n✅ Full evaluation results saved to {csv_filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', default='eval', choices=['train', 'eval'],
                        help='train: train PRAF-JSCC | eval: evaluate model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to a pretrained checkpoint to load before training')
    parser.add_argument('--model_dir', default='model_praf/', help='directory to save trained models')
    parser.add_argument('--loss_dir', default='loss_praf/', help='directory to save training loss logs')
    parser.add_argument('--eval_dir', default='result_praf/', help='directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--nf', type=int, default=256)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--snr_low_train', type=int, default=0)
    parser.add_argument('--snr_up_train', type=int, default=12)
    parser.add_argument('--snr_low_eval', type=int, default=0)
    parser.add_argument('--snr_up_eval', type=int, default=12)
    parser.add_argument('--channel_type', type=str, default='awgn',
                        help='awgn/slow_fading/slow_fading_eq/burst')
    parser.add_argument('--train_hr_dir', type=str, default='',
                        help='Path to training HR images')
    parser.add_argument('--train_lr_dir', type=str, default='',
                        help='Path to training LR images')
    parser.add_argument('--val_hr_dir', type=str, default='',
                        help='Path to validation HR images')
    parser.add_argument('--val_lr_dir', type=str, default='',
                        help='Path to validation LR images')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PRAF_JSCC(nf=args.nf, c=8, channel_type=args.channel_type).to(device)

    if args.command == 'train':
        train_praf_jscc(args, model, device)

    elif args.command == 'eval':
        model_path = os.path.join(args.model_dir,
                                  f"last_praf_jscc_snr{args.snr_low_train}to{args.snr_up_train}_{args.channel_type}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")

        evaluate_praf_jscc(args, model, device)

if __name__ == '__main__':
    main()

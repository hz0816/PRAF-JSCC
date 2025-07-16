import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import torchvision.transforms as transforms


class dataset(Dataset):
    def __init__(self, hr_dir, lr_dir, snr_range=(0, 20), patch_size=None, augment=True):
        self.hr_images = sorted(glob(os.path.join(hr_dir, "*.png")))
        self.lr_images = sorted(glob(os.path.join(lr_dir, "*.png")))

        self.snr_range = snr_range
        self.patch_size = patch_size
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_images[idx]).convert("RGB")
        lr_img = Image.open(self.lr_images[idx]).convert("RGB")
        if hr_img.width % lr_img.width != 0 or hr_img.height % lr_img.height != 0:
            return self.__getitem__(idx + 1)

        if self.patch_size is not None:
            if self.augment:
                hr_img, lr_img = self._random_crop(hr_img, lr_img, self.patch_size)
            else:
                hr_img, lr_img = self._center_crop(hr_img, lr_img, self.patch_size)
        if self.augment:
            hr_img, lr_img = self._augment(hr_img, lr_img)

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        snr = random.uniform(*self.snr_range)
        snr_tensor = torch.tensor([snr], dtype=torch.float32)

        return lr_tensor, hr_tensor, snr_tensor

    def _center_crop(self, hr_img, lr_img, patch_size):
        hr_w, hr_h = hr_img.size
        scale = hr_w // lr_img.size[0]
        lr_patch = patch_size // scale

        # 计算中心坐标
        lr_x = (lr_img.size[0] - lr_patch) // 2
        lr_y = (lr_img.size[1] - lr_patch) // 2
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_patch, lr_y + lr_patch))
        hr_crop = hr_img.crop((hr_x, hr_y, hr_x + patch_size, hr_y + patch_size))
        return hr_crop, lr_crop

    def _random_crop(self, hr_img, lr_img, patch_size):

        hr_w, hr_h = hr_img.size
        scale = hr_w // lr_img.size[0]
        lr_patch = patch_size // scale

        lr_x = random.randint(0, lr_img.size[0] - lr_patch)
        lr_y = random.randint(0, lr_img.size[1] - lr_patch)
        hr_x = lr_x * scale
        hr_y = lr_y * scale

        lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_patch, lr_y + lr_patch))
        hr_crop = hr_img.crop((hr_x, hr_y, hr_x + patch_size, hr_y + patch_size))
        return hr_crop, lr_crop

    def _augment(self, hr_img, lr_img):

        if random.random() < 0.5:
            hr_img = hr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            lr_img = lr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            hr_img = hr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            lr_img = lr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        rot_times = random.randint(0, 3)
        if rot_times > 0:
            hr_img = hr_img.rotate(90 * rot_times)
            lr_img = lr_img.rotate(90 * rot_times)

        return hr_img, lr_img

def get_snr_range(train_hr, train_lr, val_hr, val_lr, snr_low, snr_high, patch_size=128):
    train_dataset = dataset(
        hr_dir=train_hr,
        lr_dir=train_lr,
        snr_range=(snr_low, snr_high),
        patch_size=patch_size,
        augment=True
    )
    val_dataset = dataset(
        hr_dir=val_hr,
        lr_dir=val_lr,
        snr_range=(snr_low, snr_high),
        patch_size=patch_size,
        augment=False
    )
    return train_dataset, val_dataset



def get_fixed_snr(hr_dir_val, lr_dir_val, snr, patch_size=None):
    test_dataset = dataset(
        hr_dir=hr_dir_val,
        lr_dir=lr_dir_val,
        snr_range=(snr, snr),
        patch_size=patch_size,
        augment=False
    )
    return test_dataset, None

if __name__ == '__main__':
    get_fixed_snr()

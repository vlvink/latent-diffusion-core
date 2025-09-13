import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VaeDataset(Dataset):
    def __init__(self, data_dir="data", mode="train"):
        super().__init__()
        self.mode = mode

        self.paths = self._get_image_paths(data_dir, mode)

        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _get_image_paths(self, data_dir, mode):
        if mode == "train":
            train_paths = glob.glob(f"{data_dir}/train_images/*")
            val_paths = glob.glob(f"{data_dir}/val_images/*")
            paths = train_paths + val_paths
        elif mode == "val":
            paths = glob.glob(f"{data_dir}/val_images/*")
        elif mode == "test":
            paths = glob.glob(f"{data_dir}/test_images/*")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not paths:
            raise RuntimeError(f"No images found for mode '{mode}' in '{data_dir}'")
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transforms(image)


class TestVaeDataset(Dataset):
    def __init__(self, size=1000, mode="train", image_size=(256, 256)):
        super().__init__()
        self.mode = mode
        self.size = size
        self.image_size = image_size

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.rand(3, *self.image_size)

        image = (image - self.mean) / self.std
        return image
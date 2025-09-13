import glob

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
            train_paths = glob.glob(f"{data_dir}/train2017/*")
            test_paths = glob.glob(f"{data_dir}/test2017/*")
            paths = train_paths + test_paths
        elif mode == "val":
            paths = glob.glob(f"{data_dir}/val2017/*")
        elif mode == "test":
            paths = glob.glob(f"{data_dir}/test2017/*")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if not paths:
            raise RuntimeError(f"No images found for mode '{mode}' in '{data_dir}'")

        print(f"Using {len(paths)} images for {mode}")
        return paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transforms(image)

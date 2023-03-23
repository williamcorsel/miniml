from lightning.pytorch import LightningDataModule
from torchvision.datasets import CIFAR10
from torchvision import transforms as tf
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from PIL import Image

class CIFAR10Module(LightningDataModule):
    def __init__(self, data_dir="data", batch_size=32, num_workers=4, val_split=[0.8, 0.2]):
        super().__init__()
        self.save_hyperparameters()

        self.train_dir = Path(data_dir) / "train"
        self.test_dir = Path(data_dir) / "test"

        self.transforms = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def prepare_data(self):
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)


    def to_disk(self, split):
        path = Path(self.hparams.data_dir) / split
        if not path.exists():
            path.mkdir()

            if split == "train":
                dataset = self.train_dataset if hasattr(self, "train_dataset") else CIFAR10(self.hparams.data_dir, train=True, transform=self.transforms)
            elif split == "test":
                dataset = self.test_dataset if hasattr(self, "test_dataset") else CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)

            for i, (img, label) in enumerate(tqdm(dataset, desc=f"Writing images to {path}")):
                save_image(img, path / f"{i:06d}.png")
        else:
            print(f"Images already exist at {path}")

      
    def setup(self, stage):
        if stage == "fit":
            dataset = CIFAR10(self.hparams.data_dir, train=True, transform=self.transforms)
            self.train_dataset, self.val_dataset = random_split(dataset, self.hparams.val_split)
        
        if stage in ["test", "predict"]:
            self.test_dataset = CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return test_loader
    
    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )
        return predict_loader
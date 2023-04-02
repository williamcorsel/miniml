import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets.utils import download_and_extract_archive


class Craft3DModule(LightningDataModule):
    URL = "https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz"
    DATASET_FILE = "houses.tar.gz"
    SPLIT_FILE = "splits.json"

    def __init__(self, data_dir="data/Craft3D", batch_size=32, num_workers=4, block_size=32):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # Download and extract data
        file_path = Path(self.hparams.data_dir) / self.DATASET_FILE
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(self.URL, self.hparams.data_dir, filename=self.DATASET_FILE)
        
        split_path = Path(self.hparams.data_dir) / self.SPLIT_FILE
        self.splits = json.load(split_path.open())

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = self._load_schemetics("train")
            self.val_dataset = self._load_schemetics("val")
        
        if stage in ["test", "predict"]:
            self.test_dataset = self._load_schemetics("test")


    def _load_schemetics(self, split):
        schematics = []
        for filename in self.splits["train"]:
            schematic_path = Path(self.hparams.data_dir, "houses", filename, "schematic.npy")

            try:
                schematic = np.load(schematic_path)
            except FileNotFoundError as e:
                print(f"Failed to load schematic {schematic_path}")
                continue
        
            schematic = self._prepare_schematic(schematic)
            if schematic is not None:
                schematics.append(schematic)
        
        return TensorDataset(torch.from_numpy(np.stack(schematics)))
            
    def _resize_schematic(self, schematic):
        """Centers the schematic around the center of mass, resizes it to the block size and pads with zeroes.

        Args:
            schematic (np.ndarray): The schematic - of format: (y, z, x, entryshape)

        Returns:
            np.ndarray: the re-centered schematic
        """
        if schematic.sum() <= 0:
            return None
        
        # Center the schematic around the center of mass
        coords = np.array(schematic.nonzero())
        center_of_mass = np.mean(coords, axis=1)
        center_of_mass = np.round(center_of_mass).astype(int)
        schematic = np.roll(schematic, -center_of_mass, axis=(0, 1, 2))

        # Resize the schematic to the block size
        if schematic.shape[0] > self.hparams.block_size:
            schematic = schematic[: self.hparams.block_size, :, :]
        if schematic.shape[1] > self.hparams.block_size:
            schematic = schematic[:, : self.hparams.block_size, :]
        if schematic.shape[2] > self.hparams.block_size:
            schematic = schematic[:, :, : self.hparams.block_size]

        # Pad with zeroes
        if schematic.shape[0] < self.hparams.block_size:
            schematic = np.pad(
                schematic,
                ((0, self.hparams.block_size - schematic.shape[0]), (0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        if schematic.shape[1] < self.hparams.block_size:
            schematic = np.pad(
                schematic,
                ((0, 0), (0, self.hparams.block_size - schematic.shape[1]), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        if schematic.shape[2] < self.hparams.block_size:
            schematic = np.pad(
                schematic,
                ((0, 0), (0, 0), (0, self.hparams.block_size - schematic.shape[2])),
                mode="constant",
                constant_values=0,
            )

        return schematic

    def _prepare_schematic(self, schematic):
        schematic = schematic[:, :, :, 0] # Only take first (block_id)
        schematic = (schematic > 0).astype(np.uint8) # Convert to binary
        schematic = self._resize_schematic(schematic)

        if schematic is not None and schematic.sum() > 10:
            return torch.FloatTensor(schematic)
        return None
    
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
    
    def visualise(self, batch):
        """Visualise a batch of schematics in 3D using matplotlib."""
        for i, schematic in enumerate(batch):
            print(schematic.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.voxels(schematic, edgecolor="k")
            plt.show()

    

if __name__ == "__main__":
    datamodule = Craft3DModule(batch_size=5, num_workers=4, val_split=[0.8, 0.2])
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    dataloader = datamodule.train_dataloader()

    for (batch,) in dataloader:
        # print(len(batch))
        datamodule.visualise(batch)
        break
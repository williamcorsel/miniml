from lightning.pytorch import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class FiftyOneClassificationDataset(Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.
    
    Args:
        fiftyone_dataset: a FiftyOne dataset or view that will be used for training or testing
        transforms (None): a list of PyTorch transforms to apply to images and targets when loading
        gt_field ("ground_truth"): the name of the field in fiftyone_dataset that contains the 
            desired labels to load
        classes (None): a list of class strings that are used to define the mapping between
            class names and indices. If None, it will use all classes present in the given fiftyone_dataset.
    """

    def __init__(
        self,
        fiftyone_dataset,
        transforms=None,
        gt_field="ground_truth",
        classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field

        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            # Get list of distinct labels that exist in the view
            self.classes = self.samples.distinct(
                "%s.label" % gt_field
            )

        # if self.classes[0] != "background":
        #     self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path).convert("RGB")
        target = self.labels_map_rev[sample[self.gt_field]["label"]]        

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes
    

class FiftyOneDataModule(LightningDataModule):
    def __init__(self, train, val=None, test=None, val_split=None, transforms=None, batch_size=4, num_workers=4):
        super().__init__()
        self.fo_train = train
        self.fo_val = val
        self.fo_test = test
        self.val_split = val_split
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = FiftyOneClassificationDataset(self.fo_train, transforms=self.transforms)

            if self.fo_val is not None:
                self.val_dataset = FiftyOneClassificationDataset(self.fo_val, transforms=self.transforms)
            elif self.val_split is not None:
                self.train_dataset, self.val_dataset = random_split(self.train_dataset, self.val_split)
        
        if stage == "test":
            self.test_dataset = FiftyOneClassificationDataset(self.fo_test, transforms=self.transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ) if self.val_dataset is not None else None
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader
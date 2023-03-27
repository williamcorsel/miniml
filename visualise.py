import fiftyone as fo
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
from lightning.pytorch.cli import LightningCLI

from image_classification.models.modules import *
from image_classification.datasets.modules import *

def to_fiftyone(dataset, dataset_dir, name, preds=None):
    samples = []
    classes = dataset.classes
    
    for i, (img, label) in enumerate(tqdm(dataset, desc=f"Writing images to {dataset_dir}")):    
        file_path = Path(dataset_dir) / f"{i:06d}.png"

        if not file_path.exists():
            save_image(img, file_path)
            
        sample = fo.Sample(
            filepath=file_path,
            ground_truth=fo.Classification(label=classes[int(label)]),
            prediction=fo.Classification(label=classes[int(preds[i])]),

        )
        samples.append(sample)
    fo_dataset = fo.Dataset(name)
    fo_dataset.add_samples(samples)
    return fo_dataset


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
        parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
        parser.add_argument("--name", type=str, required=True, help="Dataset name")

def main():
    cli = CLI(run=False)
   
    preds = cli.trainer.predict(model=cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    preds = [p for batch in preds for p in batch]

    dataset = to_fiftyone(cli.datamodule.test_dataset, cli.config.data_dir, cli.config.name, preds)

    session = fo.launch_app(dataset)
    session.wait()

if __name__ == "__main__":
    main()

from lightning.pytorch import Trainer
from image_classification import LeNetModule, CIFAR10Module
import fiftyone as fo
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path



def to_fiftyone(dataset, dataset_dir, preds, name="cifar10"):
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


if __name__ == "__main__":
    lit_model = LeNetModule(
        num_classes=10,
    )

    data = CIFAR10Module()

    trainer = Trainer(
        max_epochs=1,
    )
    trainer.fit(
        model=lit_model,
        datamodule=data
    )

    trainer.test(
        model=lit_model,
        datamodule=data
    )
    
    preds = trainer.predict(
        model=lit_model,
        datamodule=data
    )

    # FiftyOne
    classes = data.test_dataset.classes
    print(classes)

    # data.to_disk("test")

    dataset = to_fiftyone(data.test_dataset, "data/test", preds, name="cifar10")

    print(dataset.head())

    session = fo.launch_app(dataset)
    session.wait()


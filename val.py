import torch
import torchvision.transforms as tf
from pathlib import Path
import fiftyone as fo
from fiftyone import zoo as foz
from tqdm import tqdm
from PIL import Image

from models import LeNet


if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = Path("models/run_20230322_203056/model_4.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model = LeNet(num_classes=10).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    model.eval()

    test_fo_set = foz.load_zoo_dataset("cifar10", split="test")
    classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    # Iterate over fiftyone dataset and add model predictions
    for sample in tqdm(test_fo_set, desc="Predicting"):
        img = Image.open(sample.filepath)
        img = transforms(img).unsqueeze(0).to(device)
        
        pred = model(img)
        _, pred = torch.max(pred.data, 1)
        sample["predictions"] = fo.Classification(
            label=classes[pred.item()],
        )
        sample.save()

    session = fo.launch_app(test_fo_set)
    session.wait()





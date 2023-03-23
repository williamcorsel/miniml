import torch
import torchvision
import torchvision.transforms as tf
import fiftyone as fo
import fiftyone.zoo as foz
from datetime import datetime
from tqdm import tqdm

from pathlib import Path

from models import LeNet
from datasets import FiftyOneClassificationDataset

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, desc="Evaluate", leave=False)):
            inputs, labels = data
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
    
    return correct / total

MODEL_SAVE_DIR = Path("models")
BATCH_SIZE = 4


transforms = tf.Compose([
    tf.ToTensor(),
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_fo_set = foz.load_zoo_dataset("cifar10", split="train")
test_fo_set = foz.load_zoo_dataset("cifar10", split="test")

train_set = FiftyOneClassificationDataset(train_fo_set, transforms=transforms)
test_set = FiftyOneClassificationDataset(test_fo_set, transforms=transforms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LeNet(num_classes=10).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(f"Starting training with {len(train_set)} samples using device {device}")

model_run = MODEL_SAVE_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

running_loss = -1
acc = -1
for epoch in (pbar := tqdm(range(5))):
    pbar.set_description(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader):.2f}, accuracy: {100 * acc:.2f}")
    running_loss = 0.0

    model.train()
    for i, data in enumerate(tqdm(train_loader, desc=f"Train", leave=False)):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    # Evaluate the model on the test set
    acc = test(model, test_loader, device)

    # Save the model after each epoch
    model_run.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(model_run / f"model_{epoch}.pth"))


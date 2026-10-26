import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
import time
from pathlib import Path
import csv

from src.cnn_v1 import CNN_v1
from src.dataloader import RUGDDataset

#consts
num_epochs = 50
learning_rate = 1e-4
batch_size = 8
results_dir = Path("results")
csv_dir = results_dir / "Train_log.csv"
image_dir = "database/RUGD_frames_database"
annot_dir = "database/RUGD_annotations_database"
header = ["Epoch", "Average loss", "Trainning time"]

#collate
def collate_fn(batch):
    images = []
    targets = []
    
    for sample in batch:
        img_float = sample["image"].astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).clone()
        
        # Annotation (H, W) -> Long tensor pre CrossEntropyLoss
        target_tensor = torch.from_numpy(sample["annotation"]).long().clone()
        
        images.append(img_tensor)
        targets.append(target_tensor)
        
    return torch.stack(images), torch.stack(targets)

#calculate_accuracy
##def accuracy(img_tensor, annotation):


results_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'}")
cpu = device.type == 'cpu'

train_dataset = RUGDDataset(image_dir, annot_dir, train_split=0.8, is_train=True, seed=1313)
test_dataset = RUGDDataset(image_dir, annot_dir, train_split=0.8, is_train=False, seed=1313)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

if len(train_dataset) == 0:
    print("No training samples!!!")
    exit()


if cpu:
    train_loader = TorchDataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
else:
    train_loader = TorchDataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,          
    pin_memory=True,        
    persistent_workers=True 
)

model = CNN_v1(n_channels=3, n_classes=25)
model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

with open(csv_dir, mode="w", encoding="utf-8", newline="") as subor:

    writer = csv.writer(subor)
    writer.writerow(header)

    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        num_batches = 0
        start_time = time.perf_counter()

        for images, targets in train_loader:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = nn.functional.cross_entropy(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        end_time = time.perf_counter()
        finish_time = end_time - start_time

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch: {epoch} | Average loss: {avg_loss:.4f}")

        writer.writerow(epoch, avg_loss, finish_time)

# 6. Uloženie modelu
torch.save(model.state_dict(), "results/rugd_CNN_v1.pt")
print("Training finished!")


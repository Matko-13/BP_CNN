import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
import time
from pathlib import Path
import csv
from datetime import datetime
from math import ceil

from src.cnn_v1 import CNN_1
from src.cnn_v1_2 import CNN_12
from src.cnn_v1_3 import CNN_13
from src.dataloader import RUGDDataset


def format_duration(seconds):
    minutes, seconds = divmod(max(0, ceil(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


#collate
def collate_fn(batch):
    images = torch.from_numpy(np.stack([sample["image"] for sample in batch]))
    targets = torch.from_numpy(np.stack([sample["annotation"] for sample in batch]))
    return images.permute(0, 3, 1, 2).contiguous(), targets


def prepare_batch(images, targets, device):
    images = images.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
    targets = targets.to(device, dtype=torch.long, non_blocking=True)
    return images, targets


def evaluate_model(model, test_loader, device):
    model.eval()

    correct_pixels = 0
    total_pixels = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, targets in test_loader:

            images, targets = prepare_batch(images, targets, device)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
                loss = nn.functional.cross_entropy(outputs, targets)

            predictions = torch.argmax(outputs, dim=1)

            correct_pixels += (predictions == targets).sum().item()

            total_pixels += targets.numel()

            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    accuracy = (correct_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    model.train()

    return accuracy, avg_loss  

def train(model_version):
    #consts
    num_epochs = 45
    learning_rate = 1e-4
    batch_size = 8 
    results_dir = Path("results")
    image_dir = "database/RUGD_frames_database"
    annot_dir = "database/RUGD_annotations_database"
    header = ["Epoch", "Average loss", "Testing loss", "Trainning time", "Accuracy"]

    results_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'}")
    cpu = device.type == 'cpu'
    if not cpu:
        # Vsetky obrazky maju rovnaky rozmer.
        torch.backends.cudnn.benchmark = True

    train_dataset = RUGDDataset(image_dir, annot_dir, train_split=0.8, is_train=True, seed=1313)
    test_dataset = RUGDDataset(image_dir, annot_dir, train_split=0.8, is_train=False, seed=1313)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("No training samples!!!")
        exit()


    loader_options = dict(
        num_workers=0 if cpu else 4,
        pin_memory=not cpu,
        persistent_workers=not cpu,
        collate_fn=collate_fn,
    )
    train_loader = TorchDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    test_loader = TorchDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    if(model_version == 1):
        model = CNN_1(n_channels=3, n_classes=25)
        csv_dir = results_dir / "Train_log_v1.csv"
    elif(model_version == 12):
        model = CNN_12(n_channels=3, n_classes=25)
        csv_dir = results_dir / "Train_log_v12.csv"
    elif(model_version == 13):
        model = CNN_13(n_channels=3, n_classes=25)
        csv_dir = results_dir / "Train_log_v13.csv"

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=not cpu)

    best_loss = float("inf")
    batches_per_epoch = len(train_loader)
    total_batches = num_epochs * batches_per_epoch
    completed_train_seconds = 0.0
    epoch_overhead_seconds = 0.0

    with open(csv_dir, mode="w", encoding="utf-8", newline="") as subor:

        writer = csv.writer(subor)
        writer.writerow(header)

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            num_batches = 0
            start_time = time.perf_counter()
            batch_end = start_time

            for images, targets in train_loader:
                data_time = time.perf_counter() - batch_end
                step_start = time.perf_counter()

                images, targets = prepare_batch(images, targets, device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, enabled=not cpu):
                    outputs = model(images)
                    loss = nn.functional.cross_entropy(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                step_time = time.perf_counter() - step_start
                num_batches += 1

                completed_batches = (epoch - 1) * batches_per_epoch + num_batches
                elapsed_train = completed_train_seconds + time.perf_counter() - start_time
                seconds_per_batch = elapsed_train / completed_batches
                epoch_remaining = (batches_per_epoch - num_batches) * seconds_per_batch
                remaining = (total_batches - completed_batches) * seconds_per_batch
                if epoch > 1:
                    remaining += (num_epochs - epoch + 1) * epoch_overhead_seconds / (epoch - 1)
                estimate_scope = "incl. validation" if epoch > 1 else "training only; validation not measured yet"
                finish_at = datetime.fromtimestamp(time.time() + remaining).astimezone()

                print(
                    f"Epoch {epoch}/{num_epochs} | Batch {num_batches}/{batches_per_epoch} | "
                    f"Data wait: {data_time:.3f}s | Train step: {step_time:.3f}s | "
                    f"Images/s: {images.size(0) / (data_time + step_time):.1f} | "
                    f"Total: {completed_batches}/{total_batches} ({100 * completed_batches / total_batches:.1f}%) | "
                    f"Epoch left: {format_duration(epoch_remaining)} | "
                    f"Total left: ~{format_duration(remaining)} | "
                    f"Finish: ~{finish_at:%d.%m.%Y %H:%M:%S %Z} ({estimate_scope})",
                    flush=True,
                )
                batch_end = time.perf_counter()

            end_time = time.perf_counter()
            finish_time = end_time - start_time
            completed_train_seconds += finish_time

            avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch: {epoch} | Average loss: {avg_loss:.4f} | Time: {finish_time:.4f}")

            overhead_start = time.perf_counter()
            acc, test_loss = evaluate_model(model, test_loader, device)

            if test_loss < best_loss:
                torch.save(model.state_dict(), f"results/rugd_CNN_v{model_version}.pt")
                best_loss = test_loss
                writer.writerow([epoch, avg_loss, test_loss, finish_time, acc, "Best model saved"])
            else:
                writer.writerow([epoch, avg_loss, test_loss, finish_time, acc])

            subor.flush()

            epoch_overhead_seconds += time.perf_counter() - overhead_start

    # 6. Uloženie modelu
        torch.save(model.state_dict(), "results/rugd_CNN_v1.pt")
        print("Training finished!")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    #train(model_version=12)
    train(model_version=13)
    #train(model_version=1)

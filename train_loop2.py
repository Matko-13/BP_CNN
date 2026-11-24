import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.fast_scnn import FSCN

from src.dataloader import RUGDDataset

N_CLASSES = 25

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def make_collate_fn(augment):
    def collate(batch):
        images, targets = [], []
        for sample in batch:
            img, ann = sample["image"], sample["annotation"]
            if augment:
                if random.random() < 0.5:                      # horizontal flip
                    img, ann = img[:, ::-1], ann[:, ::-1]
                h, w = ann.shape[:2]                           # random crop 512x512
                top, left = random.randint(0, h - 512), random.randint(0, w - 512)
                img = img[top:top + 512, left:left + 512]
                ann = ann[top:top + 512, left:left + 512]
            images.append(np.ascontiguousarray(img))
            targets.append(np.ascontiguousarray(ann))
        images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)   # B,H,W,C -> B,C,H,W
        targets = torch.from_numpy(np.stack(targets))
        return images, targets
    return collate


def to_device(images, targets, device):
    return images.to(device).float() / 255.0, targets.to(device).long()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=device)  
    loss_sum = 0.0

    for images, targets in loader:
        images, targets = to_device(images, targets, device)
        outputs = model(images)
        loss_sum += F.cross_entropy(outputs, targets).item()
        preds = outputs.argmax(1)
        conf += torch.bincount((targets * N_CLASSES + preds).flatten(),
                               minlength=N_CLASSES ** 2).view(N_CLASSES, N_CLASSES)

    model.train()

    tp = conf.diag().float()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn)            # trieda, ktora sa nevyskytla, da NaN
    f1 = 2 * tp / (2 * tp + fp + fn)

    return {
        "loss": loss_sum / len(loader),
        "accuracy": (tp.sum() / conf.sum()).item() * 100,
        "miou": iou.nanmean().item() * 100,   # nanmean preskoci chybajuce triedy
        "f1": f1.nanmean().item() * 100,
    }

def train():
    num_epochs = 45
    lr = 1e-3
    batch_size = 8
    image_dir = "database/RUGD_frames_database"
    annot_dir = "database/RUGD_annotations_database"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    train_set = RUGDDataset(image_dir, annot_dir, train_split=0.8, is_train=True, seed=1313)
    test_set = RUGDDataset(image_dir, annot_dir, train_split=0.8, is_train=False, seed=1313)
    print(f"Training samples: {len(train_set)}, testing samples: {len(test_set)}")

    workers = 0 if device.type == "cpu" else 4
    train_loader = DataLoader(train_set, batch_size, shuffle=True,
                              collate_fn=make_collate_fn(True),
                              num_workers=workers, persistent_workers=workers > 0)
    test_loader = DataLoader(test_set, batch_size, shuffle=False,
                             collate_fn=make_collate_fn(False),
                             num_workers=workers, persistent_workers=workers > 0)

    model = FSCN(N_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Path("results").mkdir(exist_ok=True)
    writer = SummaryWriter(f"results/fscn_{datetime.now():%Y%m%d_%H%M%S}")
    best_miou = -1.0

    for epoch in range(1, num_epochs + 1):
        start = time.perf_counter()
        train_loss = 0.0

        for images, targets in train_loader:
            images, targets = to_device(images, targets, device)
            loss = F.cross_entropy(model(images), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_time = time.perf_counter() - start    
        train_loss /= len(train_loader)
        m = evaluate(model, test_loader, device)

        print(f"Epoch {epoch}/{num_epochs} | train loss {train_loss:.4f} | test loss {m['loss']:.4f} | "
              f"acc {m['accuracy']:.2f}% | mIoU {m['miou']:.2f}% | F1 {m['f1']:.2f}% | "
              f"time {epoch_time:.1f}s")

        writer.add_scalars("Loss", {"train": train_loss, "test": m["loss"]}, epoch)
        writer.add_scalar("Metrics/accuracy", m["accuracy"], epoch)
        writer.add_scalar("Metrics/mIoU", m["miou"], epoch)
        writer.add_scalar("Metrics/F1", m["f1"], epoch)
        writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        if m["miou"] > best_miou:
            best_miou = m["miou"]
            torch.save(model.state_dict(), f"results/rugd_fscn_best{epoch}.pt")

    torch.save(model.state_dict(), "results/rugd_fscn_last.pt")
    writer.close()


if __name__ == "__main__":
    train()

from pathlib import Path
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import numpy as np
import time

from src.cnn_v1 import CNN_v1
from src.dataloader import RUGDDataset

RUGD_CLASSES = [
    {"id": 0, "R": 0, "G": 0, "B": 0},         # void_c
    {"id": 1, "R": 108, "G": 64, "B": 20},     # dirt
    {"id": 2, "R": 255, "G": 229, "B": 204},   # sand
    {"id": 3, "R": 0, "G": 102, "B": 0},       # grass
    {"id": 4, "R": 0, "G": 255, "B": 0},       # tree
    {"id": 5, "R": 0, "G": 153, "B": 153},     # pole
    {"id": 6, "R": 0, "G": 128, "B": 255},     # water
    {"id": 7, "R": 0, "G": 0, "B": 255},       # sky
    {"id": 8, "R": 255, "G": 255, "B": 0},     # vehicle
    {"id": 9, "R": 255, "G": 0, "B": 127},     # container
    {"id": 10, "R": 64, "G": 64, "B": 64},     # asphalt
    {"id": 11, "R": 255, "G": 128, "B": 0},    # gravel
    {"id": 12, "R": 255, "G": 0, "B": 0},      # building
    {"id": 13, "R": 153, "G": 76, "B": 0},     # mulch
    {"id": 14, "R": 102, "G": 102, "B": 0},    # rock_bed
    {"id": 15, "R": 102, "G": 0, "B": 0},      # log_c
    {"id": 16, "R": 0, "G": 255, "B": 128},    # bicycle
    {"id": 17, "R": 204, "G": 153, "B": 255},  # person
    {"id": 18, "R": 102, "G": 0, "B": 204},    # fence
    {"id": 19, "R": 255, "G": 153, "B": 204},  # bush
    {"id": 20, "R": 0, "G": 102, "B": 102},    # sign
    {"id": 21, "R": 153, "G": 204, "B": 255},  # rock
    {"id": 22, "R": 102, "G": 255, "B": 255},  # bridge
    {"id": 23, "R": 101, "G": 101, "B": 11},   # concrete
    {"id": 24, "R": 114, "G": 85, "B": 47}     # picnic_table
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'CUDA' if device.type == 'cuda' else 'CPU'}")

my_model = CNN_v1(n_channels=3, n_classes=25)
my_model.load_state_dict(torch.load('results/rugd_CNN_v1.pt', map_location=device))
my_model.eval()

image_dir = "database/RUGD_frames_database"
annot_dir = "database/RUGD_annotations_database"

dataset = RUGDDataset(image_dir, annot_dir, train_split=1, is_train=True, seed=1313)

CLASS_TO_COLOR = np.zeros((len(RUGD_CLASSES), 3), dtype=np.uint8)
for cls in RUGD_CLASSES:
    CLASS_TO_COLOR[cls["id"]] = [cls["R"], cls["G"], cls["B"]]

def prepare_single_image(img_np, device):
    tensor_img = torch.from_numpy(img_np.copy())
    tensor_img = tensor_img.permute(2, 0, 1).contiguous()
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
    
    return tensor_img


def make_set(id):
    sample = dataset[id]
    
    img = sample["image"]
    annot = sample["annotation"]

    start_time = time.time()
    
    tensor_img = prepare_single_image(img, device)

    my_model.eval()

    with torch.no_grad():
        output = my_model(tensor_img)

    predictions = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    color_mask = CLASS_TO_COLOR[predictions]
    annot_color = CLASS_TO_COLOR[annot]

    duration = float(time.time() - start_time)

    return color_mask, img, annot_color, duration

def evaluate_images(annotation, prediction):
    correct_pixels = np.all(annotation == prediction, axis=-1).sum()
    total_pixels = annotation.shape[0] * annotation.shape[1]
    accuracy = (correct_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
    return accuracy

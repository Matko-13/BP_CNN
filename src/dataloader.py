from pathlib import Path
import random
import numpy as np
import cv2
from torch.utils.data import Dataset

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

class RUGDDataset(Dataset):
    def __init__(self, image_dir: str, annotation_dir: str, train_split: float = 0.8, is_train: bool = True, seed: int = 420):
        self.samples = []
        
        img_dir_path = Path(image_dir)
        ann_dir_path = Path(annotation_dir)

        if img_dir_path.exists() and ann_dir_path.exists():
            image_paths = sorted([p for p in img_dir_path.glob("*.png")])
            
            path_pairs = []
            for img_path in image_paths:
                annotation_path = ann_dir_path / img_path.name
                if annotation_path.exists():
                    path_pairs.append((img_path, annotation_path))

            rng = random.Random(seed)
            rng.shuffle(path_pairs)

            train_size = int(train_split * len(path_pairs))
            if is_train:
                self.samples = path_pairs[:train_size]
            else:
                self.samples = path_pairs[train_size:]
        else:
            print("Error: Image or annotation directory does not exist.")

        self.t_width = 688
        self.t_height = 544

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, ann_path = self.samples[index]

        image_uncut = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        annotation_RGB_uncut = cv2.imread(str(ann_path), cv2.IMREAD_COLOR)

        # Orezanie na stred (rovnako ako v C++)
        start_x = (image_uncut.shape[1] - self.t_width) // 2
        start_y = (image_uncut.shape[0] - self.t_height) // 2

        image = image_uncut[start_y:start_y + self.t_height, start_x:start_x + self.t_width].copy()
        annotation_RGB = annotation_RGB_uncut[start_y:start_y + self.t_height, start_x:start_x + self.t_width].copy()

        # Konverzia RGB anotácie na triedy (ID)
        annotation = np.zeros((annotation_RGB.shape[0], annotation_RGB.shape[1]), dtype=np.uint8)

        for r in range(annotation_RGB.shape[0]):
            for c in range(annotation_RGB.shape[1]):
                bgr = annotation_RGB[r, c]
                b, g, r_val = bgr[0], bgr[1], bgr[2]

                assigned_id = 0
                for cls in RUGD_CLASSES:
                    if cls["B"] == b and cls["G"] == g and cls["R"] == r_val:
                        assigned_id = cls["id"]
                        break
                annotation[r, c] = assigned_id

        return {
            "image": image,
            "annotation": annotation,
            "image_path": str(img_path),
            "annotation_path": str(ann_path)
        }
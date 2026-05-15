import os
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from data import cls_to_names

class CLIPClassifierWrapper(nn.Module):
    def __init__(self, clip_model, text_features):
        super().__init__()
        self.clip_model = clip_model

        self.register_buffer("text_features", text_features)
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, images):
        # images: [0,1] float tensor
        x = (images - self.mean) / self.std          # CLIP normalize
        img_feat = self.clip_model.encode_image(x)
        img_feat = F.normalize(img_feat.float(), dim=-1)
        txt_feat = F.normalize(self.text_features.float(), dim=-1)
        return 100.0 * img_feat @ txt_feat.T

mapping = {
    "caltech101": ["101_ObjectCategories", "split_zhou_Caltech101.json", cls_to_names.caltech101_classes],
    "dtd": ["images", "split_zhou_DescribableTextures.json", cls_to_names.dtd_classes],
    "eurosat": ["2750", "split_zhou_EuroSAT.json", cls_to_names.eurosat_classes],
    "stanford_cars": ["", "split_zhou_StanfordCars.json", cls_to_names.cars_classes],
    "flowers102": ["jpg", "split_zhou_Flowers102.json", cls_to_names.flower102_classes],
    "ucf101": ["UCF-101", "split_zhou_UCF101.json", cls_to_names.ucf101_classes],
    "pets": ["images", "split_zhou_OxfordPets.json", cls_to_names.pets_classes],
    "aircraft": ["images", "split_zhou_Aircraft.json", cls_to_names.aircraft_classes],
    # Please add more datasets here if needed, following the format:
    # "dataset_name": ["image_directory", "split_json_file", class_names_list]
}


def load_datasets_with_split(dataset_root, type="caltech101", split='test'):
    img_dir = os.path.join(dataset_root, mapping[type][0])
    json_path = os.path.join(dataset_root, mapping[type][1])
    assert os.path.isdir(img_dir), f"Image directory not found: {img_dir}"
    assert os.path.isfile(json_path), f"Split file not found: {json_path}"

    with open(json_path) as f:
        splits = json.load(f)
    path_label_list = [
        (os.path.join(img_dir, s[0]), int(s[1]))
        for s in splits[split]
    ]
    return path_label_list, mapping[type][2]

class CustomDataset(Dataset):
    def __init__(self, path_label_list, transform=None, type="caltech101"):
        self.samples = path_label_list
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = Image.open(path).convert("RGB")
        tensor = self.transform(pil) if self.transform else pil
        return {
            "image":      tensor,
            "label":      torch.tensor(label, dtype=torch.long),
            "pil":        pil,
            "label_name": mapping[self.type][2][label]
        }
    
class EvalDataset(Dataset):
    def __init__(self, pil_label_list, type, transform=None):
        self.samples = pil_label_list
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pil, label = self.samples[idx]
        tensor = self.transform(pil) if self.transform else pil
        return {
            "image":      tensor,
            "label":      torch.tensor(label, dtype=torch.long),
            "pil":        pil,
            "label_name": mapping[self.type][2][label],
        }

def collate(batch):
    return {
        "image":      torch.stack([b["image"] for b in batch]),
        "label":      torch.stack([b["label"] for b in batch]),
        "pil":        [b["pil"] for b in batch],
        "label_name": [b["label_name"] for b in batch],
    }

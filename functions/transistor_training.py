import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from functions.t5 import encode_text
from functions.utils import (
    show_segmentation, 
    show_image, 
    clear_folder, 
    create_gif,
)
from functions.vqgan import (
    load_process_encode_rgb_image, 
    generate_iteration, 
    load_and_process_segmentation,
)
import config


class TextImageDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, min_dim=256):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.annotations = data['annotations']
        self.img_dir = img_dir
        self.transform = transform
        self.min_dim = min_dim

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_id = ann['image_id']
        caption = ann['caption']
        
        img_name = f"{img_id:012d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            with Image.open(img_path) as img:
                if min(img.size) < self.min_dim:
                    return None
                image = img.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return caption, image, img_path
        except (IOError, OSError):
            return None

    def visualize_sample(self, idx):
        item = self[idx]
        if item is None:
            print(f"Unable to visualize sample at index {idx}")
            return

        caption, image_tensor, _ = item
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image_tensor.clone()
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)
        
        image = image.numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.figtext(0.5, 0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()
        

class FilteredTextImageDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, min_dim=256, keywords=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.annotations = data['annotations']
        self.img_dir = img_dir
        self.transform = transform
        self.min_dim = min_dim
        self.keywords = keywords or [
            'landscape', 'nature', 'mountain', 'tree', 'river', 'forest', 'snow', 'stream',
            'valley', 'hill', 'field', 'meadow', 'prairie', 'canyon', 'cliff', 'waterfall',
            'lake', 'ocean', 'beach', 'coast', 'island', 'sunset', 'sunrise', 'cloud',
            'fog', 'mist', 'desert', 'glacier', 'volcano', 'plain', 'savanna', 'tundra',
            'grassland', 'woodland', 'fjord', 'gorge', 'plateau', 'vista', 'panorama',
        ]
        
        # Filter the dataset based on keywords
        self.filtered_annotations = self._filter_dataset()

    def _filter_dataset(self):
        filtered = []
        for ann in self.annotations:
            caption = ann['caption'].lower()
            if any(keyword in caption for keyword in self.keywords):
                filtered.append(ann)
        print(f"🔎 Filtered {len(filtered)} images out of {len(self.annotations)}")
        return filtered

    def __len__(self):
        return len(self.filtered_annotations)

    def __getitem__(self, idx):
        ann = self.filtered_annotations[idx]
        img_id = ann['image_id']
        caption = ann['caption']
        
        img_name = f"{img_id:012d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            with Image.open(img_path) as img:
                if min(img.size) < self.min_dim:
                    return self.__getitem__((idx + 1) % len(self))  # Try next image
                image = img.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return caption, image, img_path
        except (IOError, OSError):
            return self.__getitem__((idx + 1) % len(self))  # Try next image

    def visualize_sample(self, idx):
        caption, image_tensor, _ = self[idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # If the image is already normalized, denormalize it
        if image_tensor.min() < 0 or image_tensor.max() > 1:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image = image_tensor.clone()
            for t, m, s in zip(image, mean, std):
                t.mul_(s).add_(m)
        else:
            image = image_tensor
        
        # Convert to numpy and transpose
        image = image.numpy().transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.figtext(0.5, 0.05, caption, wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()
        

def collate_fn(batch):
    # Filter out None values
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

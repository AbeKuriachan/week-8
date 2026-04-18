import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MedicalImagingDataset(Dataset):
    def __init__(self, meta_csv_path, img_dir, transform=None, mode='train'):
        self.df = pd.read_csv(meta_csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        
        # In this dataset, there are 30 unlabeled images. We drop them for training.
        if mode in ['train', 'val', 'test']:
            self.df = self.df.dropna(subset=['label'])
            
            # Simple manual split for hold-out based on patient randomness
            # Note: We simulate a split. In real clinical scenarios, split by patient_id!
            np_random = np.random.RandomState(42)
            mask = np_random.rand(len(self.df))
            
            if mode == 'train':
                self.df = self.df[mask < 0.7]
            elif mode == 'val':
                self.df = self.df[(mask >= 0.7) & (mask < 0.85)]
            else: # test
                self.df = self.df[mask >= 0.85]
                
        elif mode == 'unlabeled':
            self.df = self.df[self.df['label'].isna()]

        # Encoding labels
        self.unique_labels = sorted([l for l in pd.read_csv(meta_csv_path)['label'].unique() if pd.notna(l)])
        self.label_map = {l: idx for idx, l in enumerate(self.unique_labels)}
        self.num_classes = len(self.unique_labels)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            # Fallback to noise if an image is missing during loading
            image = Image.fromarray(pd.np.random.randint(0, 256, (224, 224, 3), dtype=pd.np.uint8))
            
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'unlabeled':
            return image, img_id
        
        label_str = row['label']
        label_idx = self.label_map[label_str]
        return image, label_idx
        
def get_dataloaders(meta_csv_path, img_dir, batch_size=32):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds_train = MedicalImagingDataset(meta_csv_path, img_dir, transform=transform_train, mode='train')
    ds_val = MedicalImagingDataset(meta_csv_path, img_dir, transform=transform_test, mode='val')
    ds_test = MedicalImagingDataset(meta_csv_path, img_dir, transform=transform_test, mode='test')
    ds_unl = MedicalImagingDataset(meta_csv_path, img_dir, transform=transform_test, mode='unlabeled')
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    dl_unl = DataLoader(ds_unl, batch_size=batch_size, shuffle=False)
    
    return dl_train, dl_val, dl_test, dl_unl, ds_train.unique_labels

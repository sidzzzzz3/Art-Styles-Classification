from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Encoding_to_labels={
     0:'Academic_Art',
     1:'Art_Nouveau',
     2:'Baroque',
     3:'Expressionism',
     4:'Japanese_Art',
     5:'Neoclassicism',
     6:'Primitivism',
     7:'Realism',
     8:'Renaissance',
     9:'Rococo',
     10:'Romanticism',
     11:'Symbolism',
     12:'Western_Medieval'}

Labels_to_encoding={
     'Academic_Art':0,
     'Art_Nouveau':1,
     'Baroque':2,
     'Expressionism':3,
     'Japanese_Art':4,
     'Neoclassicism':5,
     'Primitivism':6,
     'Realism':7,
     'Renaissance':8,
     'Rococo':9,
     'Romanticism':10,
     'Symbolism':11,
     'Western_Medieval':12}

class ArtDataset(Dataset):
    def __init__(self,image_path,transform=None):
        self.image_path = image_path
        self.labels = pd.read_csv(image_path+"\labels.csv")
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        img_name = self.image_path + f"\{self.labels.iloc[idx,0]}"
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        if "Class" in self.labels.columns:
            label = self.labels.iloc[idx,1]
            label = Labels_to_encoding[label]
            
            sample = (image,label)
        else:
            sample = (image)
        
        return sample



#transform defination
class RandomNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
class RandomNoise(object):
    def __init__(self, mean=0.0, std=0.1, **kwargs):
        self.mean = mean
        self.std = std
        self.kwargs = kwargs  # Store additional keyword arguments

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
mean, std=(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
img_size=224
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((img_size,img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    RandomNoise(mean=0.0, std=0.05),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
    transforms.Normalize(mean=mean, std=std)
    ])

train_dataset = ArtDataset(image_path=r"C:\Users\15786\Desktop\ArtClass\data\train",transform=transform) 
val_dataset = ArtDataset(image_path=r"C:\Users\15786\Desktop\ArtClass\data\val",transform=transform) 
test_dataset = ArtDataset(image_path=r"C:\Users\15786\Desktop\ArtClass\data\test",transform=transform) 

#visualization
figure = plt.figure(figsize=(8,8))
cols,rows=2,2
for i in range(1,cols*rows+1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img,label=train_dataset[sample_idx]
    plt.subplot(rows, cols, i)
    plt.title(Encoding_to_labels[label])
    plt.axis('off')
    plt.imshow(img.permute(1,2,0)*0.5+0.5)
plt.show()

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
val_dataloader=DataLoader(val_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=64,shuffle=True)


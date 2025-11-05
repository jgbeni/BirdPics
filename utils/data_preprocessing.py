import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.v2 as T2

###### HDF5 Dataset Class ######

class HDF5Dataset(Dataset):
    def __init__(self, X, Y, train=False):
        self.imgs = X # np array of images
        self.img_labels = Y # np array of labels
        self.train = train # boolean indicating training or validation mode
        self.normalize = T.Normalize(mean=[0.5356, 0.5817, 0.6257],
                                 std=[0.2170, 0.2057, 0.2533]) # normalization transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Convert to torch tensor
        image_np = np.array(self.imgs[idx].transpose(2,0,1),float)
        image = torch.as_tensor(image_np).to(torch.float32)
        label = torch.as_tensor(np.array(self.img_labels[idx])).to(torch.float32)

        if self.train == True: # data augmentation
            image = T.RandomHorizontalFlip(p=0.5)(image)
            image = T.RandomVerticalFlip(p=0.5)(image)
            rand = np.random.random()
            if rand < 1/5:
                image = T.GaussianBlur(kernel_size = (5,9), sigma = (0.1 , 5))(image)
            elif rand < 2/5:
                image = T.ColorJitter(brightness=0.3,contrast=0.2,saturation=0.15,hue=0.15)(image)
            elif rand < 3/5:
                image = T.RandomRotation(85)(image)
            elif rand < 4/5:
                image = T2.GaussianNoise(mean=0.,sigma=0.1)(image) #T.RandomResizedCrop(size=(baseheight, width), scale=(0.75, 1.0))(image)
            image = self.normalize(image)
        else:
            self.normalize(image)

        return image, label
    

###### Label preprocessing ######

def prepare_labels(Y):
    Y_int = np.array([label_to_int_encoder(y.decode('utf-8')) for y in Y], dtype=int)
    return Y_int

def label_to_int_encoder(label):
    if label == 'golondrina':
        return 0
    elif label == 'vencejo':
        return 1
    elif label == 'avion':
        return 2
    else:
        raise ValueError(f"Unknown label: {label}")
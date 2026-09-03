import torch.nn as nn 
from torch.utils.data import DataLoader 
from torchvision.transforms import transforms as tfs 
import torchvision.datasets as ds 

def main(): 

    transform = tfs.Compose([
        tfs.ToTensor(), 
    ])

    train_dataset = ds.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = ds.CIFAR10(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )
    
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=False) 
    
    return train_data, test_data
    
if __name__ == '__main__':
    main() 


import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import DataLoader

def get_dataset(config):
    if config['dataset_type'] == 'MNIST':
        return torchvision.datasets.MNIST(root='./data/', train=True, download=True,
            transform=T.Compose([
                T.Resize((config['image_size'], config['image_size'])), 
                T.ToTensor()
            ])
        )
    elif config['dataset_type'] == 'CIFAR10':
        return torchvision.datasets.CIFAR10(root='./data/', train=True, download=True,
            transform=T.Compose([
                T.Resize((config['image_size'], config['image_size'])),
                T.ToTensor(), 
                T.RandomVerticalFlip(),
            ])
        )
    elif config['dataset_type'] == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(root='./data/', train=True, download=True,
            transform=T.Compose([
                T.Resize((config['image_size'], config['image_size'])),
                T.ToTensor()
            ])
        )
    else:
        raise ValueError(f"Unsupported dataset type: {config['dataset_type']}")

def create_dataloader(config):
    dataset = get_dataset(config)
    return DataLoader(dataset, 
                      batch_size=config['batch_size'], 
                      shuffle=True, 
                      drop_last=False, 
                      num_workers=config['num_workers'])
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def get_dataloaders(batch_size: int = 32):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
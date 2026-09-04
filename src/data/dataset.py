from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def get_dataloaders(
    batch_size: int = 32,
    augmentation_config: dict | None = None,
):
    
    train_transforms = []

    if augmentation_config and augmentation_config["enabled"]:

        if augmentation_config["horizontal_flip"]["enabled"]:
            train_transforms.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation_config["horizontal_flip"]["p"]
                )
            )

        if augmentation_config["random_crop"]["enabled"]:
            train_transforms.append(
                transforms.RandomCrop(
                    size=32,
                    padding=augmentation_config["random_crop"]["padding"],
                )
            )

        if augmentation_config["rotation"]["enabled"]:
            train_transforms.append(
                transforms.RandomRotation(
                    degrees=augmentation_config["rotation"]["degrees"],
                )
            )

    train_transforms.append(transforms.ToTensor())

    train_transform = transforms.Compose(train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=train_transform,
        download=True,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        transform=test_transform,
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
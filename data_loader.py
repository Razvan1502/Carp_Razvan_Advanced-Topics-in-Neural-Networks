import torch
from torchvision import datasets, transforms


def get_dataloader(dataset_name, batch_size, augmentations, cache_dir="data/"):
    transform_list = []


    for aug in augmentations:
        if aug['name'] == "RandomCrop":
            transform_list.append(transforms.RandomCrop(aug['size'], padding=aug['padding']))
        elif aug['name'] == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip())
        elif aug['name'] == "RandomVerticalFlip":
            transform_list.append(transforms.RandomVerticalFlip())
        elif aug['name'] == "ColorJitter":
            transform_list.append(transforms.ColorJitter(
                brightness=aug.get('brightness', 0),
                contrast=aug.get('contrast', 0),
                saturation=aug.get('saturation', 0),
                hue=aug.get('hue', 0)))
        elif aug['name'] == "RandomRotation":
            transform_list.append(transforms.RandomRotation(degrees=aug['degrees']))

    # Add ToTensor before RandomErasing
    transform_list.append(transforms.ToTensor())

    # Add RandomErasing after ToTensor
    for aug in augmentations:
        if aug['name'] == "RandomErasing":
            transform_list.append(transforms.RandomErasing(
                p=aug.get('p', 0.5),
                scale=aug.get('scale', [0.02, 0.33]),
                ratio=aug.get('ratio', [0.3, 3.3])))

    # Compose all transforms
    transform = transforms.Compose(transform_list)


    if dataset_name == "MNIST":
        dataset = datasets.MNIST(cache_dir, train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR-10":
        dataset = datasets.CIFAR10(cache_dir, train=True, download=True, transform=transform)
    elif dataset_name == "CIFAR-100":
        dataset = datasets.CIFAR100(cache_dir, train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader, train_loader

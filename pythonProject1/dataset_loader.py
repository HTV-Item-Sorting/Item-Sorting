from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


class WasteDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.class_mapping = {i: i for i in range(6)}  # Assuming classes are already 0-5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        # Assume the label is already a number from 0 to 5
        label = item['objects']['category'][0] if item['objects']['category'] else 0

        if self.transform:
            image = self.transform(image)

        return image, label


def load_waste_dataset(dataset_name):
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create WasteDataset for each split
    train_dataset = WasteDataset(dataset['train'], transform=transform)
    val_dataset = WasteDataset(dataset['validation'] if 'validation' in dataset else dataset['test'],
                               transform=transform)

    return train_dataset, val_dataset


def create_data_loaders(dataset_name, batch_size=32):
    train_dataset, val_dataset = load_waste_dataset(dataset_name)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


# Example usage
dataset_name = 'keremberke/garbage-object-detection'  # Replace with actual dataset name
train_loader, val_loader = create_data_loaders(dataset_name)

# Print some information about the dataset
print(f"Number of training samples: {len(train_loader.dataset)}")
print(f"Number of validation samples: {len(val_loader.dataset)}")

# Get class names (labels)
class_names = list(train_loader.dataset.class_mapping.keys())
print(f"Classes: {class_names}")

# Example of iterating through the data
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
    break  # Just print the first batch
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataLoader:
    """
    MNIST dataset loader with preprocessing and data augmentation.
    """
    
    def __init__(self, batch_size=128, validation_split=0.1, data_dir='./data'):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.data_dir = data_dir
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation up to 10 degrees
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def get_data_loaders(self):
        """
        Returns train, validation, and test data loaders.
        """
        # Load training dataset
        full_train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        # Load test dataset
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Split training data into train and validation
        train_size = int((1 - self.validation_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create validation dataset with test transforms (no augmentation)
        val_dataset.dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def visualize_samples(self, data_loader, num_samples=8):
        """
        Visualize sample images from the dataset.
        """
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Denormalize images for visualization
        mean = 0.1307
        std = 0.3081
        images = images * std + mean
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = images[i].squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {labels[i].item()}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_distribution(self, data_loader):
        """
        Get the distribution of classes in the dataset.
        """
        class_counts = torch.zeros(10)
        
        for _, labels in data_loader:
            for label in labels:
                class_counts[label] += 1
        
        return class_counts.numpy()
    
    def print_dataset_info(self, train_loader, val_loader, test_loader):
        """
        Print information about the datasets.
        """
        print("Dataset Information:")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of classes: 10")
        
        # Print class distribution for training set
        train_dist = self.get_class_distribution(train_loader)
        print("\nTraining set class distribution:")
        for i, count in enumerate(train_dist):
            print(f"Class {i}: {int(count)} samples ({count/len(train_loader.dataset)*100:.1f}%)")

if __name__ == "__main__":
    # Test the data loader
    data_loader = MNISTDataLoader(batch_size=64)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    
    data_loader.print_dataset_info(train_loader, val_loader, test_loader)
    data_loader.visualize_samples(train_loader)
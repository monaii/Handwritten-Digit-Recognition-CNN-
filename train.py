import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os

from model import DigitCNN
from data_loader import MNISTDataLoader

class Trainer:
    """
    Training class for the handwritten digit recognition CNN.
    """
    
    def __init__(self, model, device, train_loader, val_loader, test_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            accuracy = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, criterion):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= len(self.val_loader)
        val_accuracy = 100. * correct / total
        
        return val_loss, val_accuracy
    
    def test(self):
        """Test the model on test set."""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Per-class accuracy
                c = (pred.squeeze() == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        test_accuracy = 100. * correct / total
        
        print(f'\nTest Accuracy: {test_accuracy:.2f}% ({correct}/{total})')
        print('\nPer-class accuracy:')
        for i in range(10):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f'Class {i}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
        
        return test_accuracy
    
    def train(self, epochs=20, learning_rate=0.001, weight_decay=1e-4):
        """Main training loop."""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f'New best validation accuracy: {val_acc:.2f}%')
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping if we reach 99% validation accuracy
            if val_acc >= 99.0:
                print(f'\nReached 99% validation accuracy! Stopping early.')
                break
        
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time:.2f} seconds')
        print(f'Best validation accuracy: {self.best_val_accuracy:.2f}%')
        
        # Load best model for testing
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.best_val_accuracy
    
    def plot_training_history(self):
        """Plot training and validation metrics."""
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_predictions(self, num_samples=8):
        """Visualize model predictions on test samples."""
        self.model.eval()
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            predictions = outputs.argmax(dim=1)
        
        # Move back to CPU for visualization
        images = images.cpu()
        predictions = predictions.cpu()
        
        # Denormalize images
        mean = 0.1307
        std = 0.3081
        images = images * std + mean
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            img = images[i].squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='best_model.pth'):
        """Save the best model."""
        if self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'best_val_accuracy': self.best_val_accuracy,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies
            }, filepath)
            print(f'Model saved to {filepath}')

def main():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading MNIST dataset...")
    data_loader = MNISTDataLoader(batch_size=128, validation_split=0.1)
    train_loader, val_loader, test_loader = data_loader.get_data_loaders()
    data_loader.print_dataset_info(train_loader, val_loader, test_loader)
    
    # Create model
    print("\nCreating model...")
    model = DigitCNN(dropout_rate=0.25)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create trainer
    trainer = Trainer(model, device, train_loader, val_loader, test_loader)
    
    # Train model
    best_accuracy = trainer.train(epochs=30, learning_rate=0.001)
    
    # Test model
    print("\nTesting final model...")
    test_accuracy = trainer.test()
    
    # Visualizations
    print("\nGenerating visualizations...")
    trainer.plot_training_history()
    trainer.visualize_predictions()
    
    # Save model
    trainer.save_model('digit_recognition_cnn.pth')
    
    print(f"\nFinal Results:")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    if best_accuracy >= 99.0:
        print("🎉 Successfully achieved 99%+ validation accuracy!")
    else:
        print("Target of 99% validation accuracy not reached. Consider:")
        print("- Increasing number of epochs")
        print("- Adjusting learning rate")
        print("- Modifying model architecture")
        print("- Adding more data augmentation")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms

# Define a simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Two convolutional layers followed by three fully-connected layers
        self.conv1 = nn.Conv2d(3, 6, 5)      # input: 3 channels, output: 6 channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)         # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)       # input: 6 channels, output: 16 channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)           # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 -> relu -> pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 -> relu -> pool
        x = x.view(-1, 16 * 5 * 5)            # flatten the tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to train the model over multiple epochs and batches
def train_model(rank, model, num_epochs, batch_size, lock):
    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Define transformations and load the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training loop over epochs and batches
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(trainloader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()          # Clear previous gradients
            outputs = model(inputs)        # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()                # Backward pass (compute gradients)
            
            # Use lock to ensure that optimizer.step() is executed by one process at a time
            with lock:
                optimizer.step()           # Update model parameters
                
            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Process {rank} | Epoch {epoch + 1} | Batch {i + 1} | Loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    print(f"Process {rank} finished training.")

if __name__ == '__main__':
    # Set the multiprocessing start method to 'spawn' (suitable for CUDA and cross-platform)
    mp.set_start_method("spawn")
    
    num_processes = 6   # Number of processes to run concurrently
    num_epochs = 10      # Number of training epochs
    batch_size = 128     # Batch size for training
    
    # Create the model and move its parameters to shared memory so all processes share the same model
    model = SimpleCNN()
    model.share_memory()
    
    # Create a lock to synchronize updates to the model parameters
    lock = mp.Lock()
    
    # Spawn multiple training processes
    processes = [mp.Process(target=train_model, args=(i, model, num_epochs, batch_size, lock))
                 for i in range(num_processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print("Training complete.")


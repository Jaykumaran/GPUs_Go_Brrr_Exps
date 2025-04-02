import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import os
import time

# Define a simple CNN for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)   # 3 input channels (RGB), 6 output channels, 5x5 kernel
        self.pool = nn.MaxPool2d(2, 2)      # 2x2 max pooling
        self.conv2 = nn.Conv2d(6, 16, 5)    # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)        # 10 classes for CIFAR-10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool
        x = x.view(-1, 16 * 5 * 5)            # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(rank, model, epochs, batch_size):
    print(f"Process {rank} starting training with PID: {os.getpid()}, Process Name: {mp.current_process().name}")
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset (each process loads its own copy)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8)  # no extra subprocesses here

    # Set device (all processes share the same GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop for the given number of epochs
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Process {rank} | Epoch {epoch+1} | Batch {i+1}: Loss = {running_loss / 100:.3f}')
                running_loss = 0.0

    print(f"Process {rank} finished training.")

def main():
    # Use the 'spawn' start method for safety with CUDA tensors
    mp.set_start_method('spawn')
    num_processes = 1   # Number of training processes
    epochs = 10         # For demonstration, using one epoch
    batch_size = 128
    processes = []

    # Create the model and move its parameters to shared memory
    model = SimpleCNN()
    model.share_memory()  # All processes share the same model parameters

    # Start training processes
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(rank, model, epochs, batch_size))
        p.start()
        processes.append(p)
        print(f"[Main] Started training process with PID: {p.pid}")

    # Monitor active processes
    while any(p.is_alive() for p in processes):
        active = [p.pid for p in mp.active_children()]
        print(f"[Main] Active processes: {active}")
        time.sleep(1)

    # Join all processes
    for p in processes:
        p.join()
    print("[Main] All processes have finished training.")

if __name__ == '__main__':
    main()

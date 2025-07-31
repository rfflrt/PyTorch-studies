import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# ======================== Classifier for CIFAR10 ======================== #

# Transforms images to tensor and normalizes
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    )

batch_size = 4

device = 'cuda' if torch.accelerator.is_available() else 'cpu'

## Accessing and loading datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='.data/', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

## Classes to identify
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# ---- Showing images ---- #
def imshow(img):
    img = img / 2 + 0.5 # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Getting and showing images
#dataiter = iter(trainloader)
#images, labels = next(dataiter)

# Showing images
#imshow(torchvision.utils.make_grid(images))
# Print labels
#print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# ------- Convolutional Neural Network ------- #

# CNN structure
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 3 channel input, 6 channel output, 5x5 convolutions
        self.conv1 = nn.Conv2d(3, 6, 5)

        # 2x2 grid maxpooling
        self.pool = nn.MaxPool2d(2, 2)

        # 6 channel input, 16 channel output, 5x5 convolutions
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully-connected stage
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 2x2 Maxpooling over ReLU activated conv1 over input x
        x = self.pool(F.relu(self.conv1(x)))

        # 2x2 Maxpooling over ReLU activated conv1 over x
        x = self.pool(F.relu(self.conv2(x)))

        # FC stage
        x = torch.flatten(x, 1) # Flatten all except batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net().to(device)
print(f'CNN structure:\n{net}\n')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

option = int(input("Training (0) | Testing (1):\n"))

# Training script
if option == 0:
    epochs = 5
    print(f'Using {device} device')

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Data is a list of inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward + backward + optimization
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Finished training")
    PATH = './models/cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    print("Model weights saved as 'cifar_net.pth'")

# Testing script
else:
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    # Load weights
    PATH = './models/cifar_net.pth'
    net.load_state_dict(torch.load(PATH, weights_only=True))

    # Moves data to proper device
    images, labels = images.to(device), labels.to(device)
    
    # Calculates only forward pass
    with torch.no_grad():
        outputs = net(images)

    # Infers class
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
    
    ## Measure trained model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
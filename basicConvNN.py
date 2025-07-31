import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# This happens to be LeNet

## Convolutional Neural Network class

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # 1 channel input, 6 channel output, 5x5 convolution
        self.conv1 = nn.Conv2d(1,6,5)

        # 6 channel input, 16 channel output, 5x5 convolution
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Fully-connected stage
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    # ConvNN construction
    def forward(self, input):

        # Conv. layer C1: 1 channel input, 6 channel output,
        # 5x5 convolution through ReLU activation
        # Outputs tensor(N, 6, 28, 28), N = batch_size
        c1 = F.relu(self.conv1(input))

        # Subsampling layer S2: 2x2 grid
        # Outputs tensor(N, 6, 14, 140
        s2 = F.max_pool2d(c1, (2,2))

        # Conv. layer C3: 6 channels input, 16 channel output,
        # 5x5 convolution through ReLU activation
        # Outputs tensor(N, 16, 10, 100
        c3 = F.relu(self.conv2(s2))

        # Subsampling layer S4: 2x2 grid
        # Outputs tensor(N, 16, 5, 5)
        s4 = F.max_pool2d(c3, 2)

        # Flattening operation
        # Outputs tensor(N, 400)
        s4 = torch.flatten(s4, 1)

        # Fully-connected layer F5: tensor(N, 400) input,
        # tensor(N, 120) output through ReLU activation
        f5 = F.relu(self.fc1(s4))

        # Fully-connected layer F6: tensor(N, 120) input,
        # tensor(N, 84) output through ReLU activation
        f6 = F.relu(self.fc2(f5))

        # Gaussian output layer: tensor(N, 84) input,
        # tensor(N, 10) output
        output = self.fc3(f6)

        return output

net = Net()

# ------------- Printing the structure ------------- #
print("Basic Convolutional Neural Network Structure\n"\
      "Using ReLU activations and maxpooling:\n")
print(net)

## Prameters list
params = list(net.parameters())
print(f"\nThe network has {len(params)} weight matrices")
print(f"conv1 layer weight matrix is of the form\n{params[0].size()}\n")

## Example 32x32 input:
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(f"Random 32x32 input:\n{input}\n\n"\
      f"Net output:\n{out}\n")

#net.zero_grad()                 # zero gradients
#out.backward(torch.randn(1,10)) # random 10 dimensional gradient vector

# ------------- Loss Function ------------- #
# Using MSE as loss over a dummy target

target = torch.randn(1, 10) # dummy target
criterion = nn.MSELoss()
loss = criterion(out, target)
print(f"[Using MSE as loss in this example]\nFor a dummy target, we have the loss tensor:")
print(loss)
print()

# ------------- Backpropagation ------------- #

# Zero gradients to avoid accumalation
#net.zero_grad()

#print("conv1 bias gradient before backward propagation:")
#print(net.conv1.bias.grad)
#print()

#loss.backward() # Calculates gradients

#print("conv1 bias gradient after backward propagation:")
#print(net.conv1.bias.grad)

# ------------- Weight Optimization ------------- #

# Creating optimizer object
# lr = learning rate
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# In the training loop #
optimizer.zero_grad()  # zeroes the gradients
out = net(input)
loss = criterion(out, target) # Calculates the loss
loss.backward()  # Calculates gradients
optimizer.step() # Updates the weights
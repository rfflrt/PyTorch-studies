import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):

    ## Defines NN
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        ## Network structure
        self.linear_relu_stack = nn.Sequential(
            ## nn.Linear(x, y) represents a layer that takes
            #  x-d input, outputs y-d after a linear combo.
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    ## Forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

## Instantiate and visualize model
model = NeuralNetwork().to(device)
print(model)
print()

#-------------------- Step-by-step NN --------------------#

print("------------------------ Step-by-step NN (not the instantiated model) ------------------------")

## Minibatch of 3 random 28x28 images
input_image = torch.rand(3, 28, 28)
print(f"Original random_input size: {input_image.size()}")

## Flattens each image
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(f"Flattaned size: {flat_image.size()}")
print()

## First linear layer
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(f"Resulting tensor after first hidden layer (28*28 -> 20): {hidden1.size()}")
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1) # Performs ReLU
print(f"After ReLU: {hidden1}\n\n")

## Building automatic sequential pipeline
print("Sequential pipeline containing:\n" \
"- flattening\n" \
"- Linear 28*28 -> 20 |\n" \
"- ReLU               | One hidden layer\n" \
"- Linear 20 -> 10")
seq_models = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20,10)
)


## Initial random minibatch of 3 28x28 images
input_image = torch.rand(3,28,28)
print(f"Initial tensor (minibatch of 3 28x28 image):\n{input_image}\n")

## Passing the minibatch through the network
logits = seq_models(input_image)
print(f"Final tensor of logits:\n{logits}\n\n")

## Applying softmax on the final logits to obtain the probability
# dim = 1 because we want the sum over the [3,**10**] tensor to be 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(f"Probabilities obtained through softmax on logits:\n{pred_probab}\n")

#-------------------- Exploring instantiated model structure --------------------#

print("------------------------ Exploring instantiated model structure ------------------------")

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")
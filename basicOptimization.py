import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#--------- Training basic NN for multinomial classification ---------

## Accesses FashionMNIST training data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

## Accesses FashionMNIST test data
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

## Dataloaders for the datasets
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

## Simple NN
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # Flat NN
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 28x28 input 512 output
            nn.ReLU(),             # ReLU activation
            nn.Linear(512, 512),   # 512 input and output
            nn.ReLU(),             # ReLU activation
            nn.Linear(512, 10),    # 512 input, 10 output for classification (logits)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

print("Neural Network containing:\n" \
"- flattening\n" \
"- Linear 28*28 -> 512  |\n" \
"- ReLU                _| 1st hidden layer\n" \
"- Linear 512 -> 512    |\n" \
"- ReLU                _| 2nd hidden layer\n" \
"- Linear 512 -> 10    _| Output layer\n")

# Using accelerator device if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device\n")
model = NeuralNetwork().to(device)

## Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 10

## Loss function
loss_fn = nn.CrossEntropyLoss()

## Optmizer - encapsulates training (weight updates)
# Stochastic Gradeint Descent (batch size being 64, defined above)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

## Training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Model set to training mode (for batch normalization, dropout etc.)
    model.train()
    # Traverse dataset
    for batch, (X, y) in enumerate(dataloader):
        # moves data to current device
        X, y = X.to(device), y.to(device)
        # Calculate predition and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()       # Calculates gradients
        optimizer.step()      # Updates weights
        optimizer.zero_grad() # Set gradeints to 0 (prevents double-counting)
    
    if batch % 100 == 0:
        loss, current = loss.item(), batch * batch_size + len(X)
        print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

## Testing loop
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    # Model set to evaluation mode (for batch normalization, dropout etc.)
    model.eval()

    # Disabling gradient tracking since we only want to evaluate
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

## NN training (main)
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done")

## Saving model
torch.save(model, "model.pth")
print("Model saved")
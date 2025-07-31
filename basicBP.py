import torch

print(f"[5 dimensional input of 1s and expected 3 dimensional output of 0s]\n")
x = torch.ones(5)  # input 5 dim. tensor
print(f"Input: {x}")
y = torch.zeros(3) # expected 3 dim. tensor
print(f"Expected output: {y}\n")

print(f"Initial tensors requiring gradient")
w = torch.randn(5,3, requires_grad=True) # weights
print(f"Weights (w):\n{w}\n")
b = torch.randn(3, requires_grad=True)   # bias
print(f"Bias (b):\n{b}\n")
z = torch.matmul(x, w) + b               # computed output
print(f"Computed output (z): {z}")

# binary cross-entropy loss
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}\n")

## Computing gradients
print("--- Computing gradients | Partial derivatives for the loss ---\n")
loss.backward() # calculates the gradients for tensor that require so
# <.grad> is only available for leaf nodes that require gradients
# If gradients need to be calculated more than one, pass retain_graph=True to backward() call
print(f"With respecto to w:\n{w.grad}\n")
print(f"With respect to b:\n{b.grad}\n")

## Disabling gradient tracking
print("Disabling gradient to only do forward computing")
z = torch.matmul(x, w) + b
print(f"z requiring grad: {z.requires_grad}")

# If only forward computation is wanted, disable gradient tracking using:
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(f"z requires gradient: {z.requires_grad}")

# Or using detach()
z = torch.matmul(x, w) + b
z_detached = z.detach()
print(f"z_detached requires gradient: {z_detached.requires_grad}")
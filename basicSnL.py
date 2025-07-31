import torch
import torchvision.models as models

#---------------- Saving and Loading weights ----------------#
# Learned weights are stored in "model.state_dict()"

## Saving -----------------------------------
# Loads VGG16 with pre-trained weights
model = models.vgg16(weights='IMAGENET1K_V1')

# Saves the model weights on a file in the current directory 
torch.save(model.state_dict(), 'models/model_weights.pth')

## Loading ----------------------------------
# Model to wich weights will be loaded must be of the same architecture
model = models.vgg16()

# Loads weights
# weights_only=True limits functions executed to only the necessary (best practice)
model.load_state_dict(torch.load('models/model_weights.pth', weights_only=True))

# evaluation mode
model.eval()
print(model)


#---------------- Saving and Loading models structure ----------------#

## Saving -----------------------
torch.save(model, 'models/model.pth')

## Loading
# weights_only=False because we are loading the entire model
model = torch.load('models/model.pth', weights_only=False)
print(model)
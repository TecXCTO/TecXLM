import torch
data = torch.load('model.pth', map_location='cpu')

# If 'data' is a dictionary, see what keys are available
if isinstance(data, dict):
    print("Keys found in .pth file:", data.keys())
else:
    # If it's a model object, check its attributes
    print("Model attributes:", dir(data))


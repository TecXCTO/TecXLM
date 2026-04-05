import torch
data = torch.load('model.pth', map_location='cpu')

# If 'data' is a dictionary, see what keys are available
if isinstance(data, dict):
    print("Keys found in .pth file:", data.keys())
else:
    # If it's a model object, check its attributes
    print("Model attributes:", dir(data))



# Load your model file
#checkpoint = torch.load('your_model_name.pth', map_location='cpu')
checkpoint=data
# Check if the list was saved as metadata
if 'chars' in checkpoint:
    print(checkpoint['chars'])
elif 'vocab' in checkpoint:
    print(checkpoint['vocab'])
else:
    print("The list is not in the .pth file; it is defined in your tecxlm.py script.")
    

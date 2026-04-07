import torch
import torch.nn as nn
import torch.onnx
from tecxlmtrain import TecXModel

model_path = "tecxlm/tecxmodel1.pth"

# 1. Load your trained model
model = TecXModel()
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval() # Set to evaluation mode

class ExportModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Only return the first element (logits) if it's a tuple
        ##output = self.model(x)
        ##return output[0] if isinstance(output, tuple) else output
        # Explicitly only return the first element (logits)
        logits, _ = self.model(idx) 
        return logits

# Use this wrapper for exporting
export_model = ExportModel(model)

# 2. Create dummy input (match your model's input shape, e.g., 1 image, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224) 

# 3. Export to ONNX
torch.onnx.export(export_model, dummy_input, "model.onnx", export_params=True, 
                  opset_version=12, 
                  do_constant_folding=True)
"""
torch.onnx.export(model, dummy_input, "model.onnx", 
                  export_params=True, 
                  opset_version=12, 
                  do_constant_folding=True)
"""
print("PyTorch model converted to model.onnx")

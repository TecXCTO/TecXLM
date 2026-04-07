import torch
import torch.onnx
from tecxlmtrain import TecXModel
model_path = "tecxlm/tecxmodel1.pth"
# 1. Load your trained model
#model = YourModelClass()
model = TecXModel()
#model.load_state_dict(torch.load('model_checkpoint.pth')['model_state_dict'])
model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval() # Set to evaluation mode

# 2. Create dummy input (match your model's input shape, e.g., 1 image, 3 channels, 224x224)
dummy_input = torch.randn(1, 3, 224, 224) 

# 3. Export to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", 
                  export_params=True, 
                  opset_version=12, 
                  do_constant_folding=True)
print("PyTorch model converted to model.onnx")

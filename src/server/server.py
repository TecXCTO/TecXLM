import torch
from fastapi import FastAPI
from pydantic import BaseModel
from tecxlmserve import TecXModel

# 1. Define your model architecture
##class MyModel(torch.nn.Module):
    # (Your model definition here)
    #pass

app = FastAPI()
model = None
model_path="../../tecxlm/tecxmodel1.pth"
chars=""
# 2. Use lifespan to load model once at startup
@app.on_event("startup")
async def load_model():
    global model
    model = TecXModel()
    checkpoints=torch.load(model_path, map_location="cpu")
    chars=checkpoints[chars]
    #model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.load_state_dict(checkpoints)
    model.eval()

class InputData(BaseModel):
    data: list[float]

# 3. Create prediction endpoint
@app.post("/predict")
async def predict(input: InputData):
    # Convert input to tensor
    tensor_input = torch.tensor([input.data])
    
    # Run inference
    with torch.no_grad():
        output = model(tensor_input)
    
    return {"prediction": output.tolist()}
    

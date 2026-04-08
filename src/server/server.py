import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .src.server.tecxlmserve import TecXModel
#from .tecxlmserve import TecXModel

# 1. Create a dictionary or object to store the model globally
ml_models = {}
model_path="../../tecxlm/tecxmodel1.pth"
chars=""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load the model into memory ---
    print("Loading model...")
    # Replace 'MyModelClass' with your actual model class
    model = TecXModel()
    checkpoints=torch.load(model_path, map_location="cpu")
    chars=checkpoints[chars]
    model.load_state_dict(checkpoints[state_dict])
    #model.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    
    #model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()
    
    ml_models["my_model"] = model
    
    yield  # The app runs while this is suspended
    
    # --- Shutdown: Clean up resources (if needed) ---
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(data: list[float]):
    # Access the loaded model
    model = ml_models["my_model"]
    input_tensor = torch.tensor([data])
    
    with torch.no_grad():
        output = model(input_tensor)
    
    return {"prediction": output.tolist()}

import uvicorn

# ... (all your existing app and lifespan code) ...

if __name__ == "__main__":
    # Change 'main' to the name of your python file if it is different
    #uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    uvicorn.run("app", host="127.0.0.1", port=8000, reload=True)
    
"""
uvicorn main:app --reload

uvicorn src.server.server:app --reload
python -m uvicorn src.server.server:app --reload

"""

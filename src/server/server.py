import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager

# 1. Create a dictionary or object to store the model globally
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load the model into memory ---
    print("Loading model...")
    # Replace 'MyModelClass' with your actual model class
    model = MyModelClass() 
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
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


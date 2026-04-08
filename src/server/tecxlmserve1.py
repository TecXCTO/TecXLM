import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# 1. Define the same transforms you used during training
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # 2. Read the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # 3. Preprocess and add batch dimension
    input_tensor = preprocess(image).unsqueeze(0) 
    
    # 4. Run inference
    model = ml_models["my_model"] # From your lifespan loader
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        prediction = torch.argmax(probabilities).item()

    return {"class_id": prediction, "confidence": probabilities[prediction].item()}


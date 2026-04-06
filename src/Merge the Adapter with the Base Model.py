"""
To wrap up your QLoRA journey, here is how you merge and deploy that trained model so it actually works on your Snapdragon and Raspberry Pi.
1. Merge the Adapter with the Base Model
Since QLoRA only trains a tiny "adapter" file, you must merge it back into the main model to create a single, usable file.
python
"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Meta-Llama-3-8B" # Your original base model
adapter_path = "./final_qlora_adapter"   # Where you saved your training results

# 1. Load Base Model in FP16 (Must be FP16 to merge correctly)
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 2. Load the Adapter and Merge
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# 3. Save the final merged model
merged_model.save_pretrained("./merged_model_final")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./merged_model_final")
"""
Deployment Chain for Your Hardware
Now that you have the ./merged_model_final folder, follow the path we built earlier:
For Raspberry Pi:
Convert this folder to ONNX (using the script I gave you).
Run it using ONNX Runtime.
For Snapdragon:
Convert the merged model to GGUF or TFLite.
Use the Qualcomm AI Stack or MLCC-Chat to run it on your phone.
3. Final "Best Results" Checklist
Mixed Data: Did you include 10% of your old data in the training? If not, your model might forget how to talk normally.
Prompt Template: Ensure you use the exact same prompt format (e.g., ### Instruction: ... ### Response:) during inference that you used during retraining.
Quantization: For the Pi, always use INT8; for the Snapdragon 7 series, you can even try 4-bit (INT4) to save more RAM.

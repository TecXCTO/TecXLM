"""
To retrain a model using QLoRA with the Hugging Face PEFT library, follow these steps. This method is the "gold standard" for 2026 because it allows you to fine-tune massive models (like Llama 3 or Mistral) on a single consumer GPU (like your RTX 30 series) by using 4-bit quantization. 
1. Install Required Libraries 
 First, ensure you have the modern stack for efficient training. You will need transformers for the model, peft for the adapters, bitsandbytes for 4-bit quantization, and trl for the high-level trainer. 

#pip install -q -U bitsandbytes transformers peft accelerate datasets trl

2. Configure 4-bit Quantization 
Use BitsAndBytesConfig to load the model in 4-bit NormalFloat (NF4). This reduces the memory footprint by approximately 4x compared to standard 16-bit training. 

"""
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for stability
    bnb_4bit_use_double_quant=True,
)

"""

3. Load Base Model and Tokenizer
Load your pre-trained model with the quantization config. Use device_map="auto" to automatically handle memory across your hardware. 
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B" # Example model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set padding token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)


"""
4. Configure LoRA Adapters 
Define your LoraConfig. The Rank (r) and Alpha determine how many parameters are trainable. A rank of 16 is a standard starting point for high-quality results. 
"""

from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Confirms low (~1-2%) parameter update


"""
5. Initialize Trainer and Start Training
Utilize the SFTTrainer from the TRL library for streamlined supervised fine-tuning. 
"""
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./qlora_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4, 
    num_train_epochs=3,
    fp16=True, 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=your_dataset, 
    peft_config=peft_config,
    args=training_args,
)

trainer.train()
trainer.save_model("./final_qlora_adapter")

"""
6. Deployment
Post-training, combine the saved adapter with the base model for inference using PeftModel.from_pretrained().
"""

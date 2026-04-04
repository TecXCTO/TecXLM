import torch
import torch.nn as nn


print(f"importing")
##from tecxlm.TecXModel import generate
from tecxlm import TecXModel
#import tecxlm
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text="! , . : ; ? C E F H I L M T U W X a b c d e f g h i k l m n o p q r s t u v w x y "
# here are all the unique characters that occur in this text
chars = sorted(list(set(text.replace(" ",""))))
vocab_size = len(chars)

print(''.join(chars))
print(vocab_size)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

model_path = "tecxlm/TecXLM.pth"
# model_path = Path("tecxlm") / "TecXLM.pth"
print(f"tecxmodelgen creating")
class TecXModelGen(TecXModel):
  def init():
    super.init()
    super.vocab_size=71
print(f"tecxmodelgen created")
model = TecXModel(vocab_size=71)
#model = TecXModelGen()
print(f"model veriable creating")
# torch.save(model.state_dict(),"../TecXLM.pth")
model.load_state_dict(torch.load(model_path))
print(f"model veriable creating")
model.to(device)
model.eval()
print(f"model veriable evaluated")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(f"context veriable created")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print(f"model generating completed.")
#open('more_generated.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
open('TecXLM_generating.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

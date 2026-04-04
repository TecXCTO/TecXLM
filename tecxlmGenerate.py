import torch
import torch.nn as nn


print(f"importing")
##from tecxlm.TecXModel import generate
from tecxlm import TecXModel
#import tecxlm


model_path = "tecxlm/TecXLM.pth"
# model_path = Path("tecxlm") / "TecXLM.pth"
print(f"tecxmodelgen creating")
class TecXModelGen(TecXModel):
  def init():
    super.init()
print(f"tecxmodelgen created")
model = TecXModel()
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

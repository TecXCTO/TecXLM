import torch
import torch.nn as nn
from tecxlm import TecXModel

model_path = Path("tecxlm") / "TecXLM.pth"

#model = TecXModel().generate()
print(tecxmodelgen creating)
class TecXModelGen(TecXModel):
  def init():
    super.init()
print(tecxmodelgen created)
model = TecXModelGen()
print(model veriable creating)
# torch.save(model.state_dict(),"../TecXLM.pth")
model.load_state_dict(torch.load(model_path))
print(model veriable creating)
model.to(device)
model.eval()
print(model veriable evaluated)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(context veriable created)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
print(model generating completed.)
#open('more_generated.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))
open('TecXLM_generating.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

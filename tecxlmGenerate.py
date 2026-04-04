import torch
import torch.nn as nn
from tecxlm import TecXModel

model_path = Path("tecxlm") / "TecXLM.pth"

model = TecXModel()
# torch.save(model.state_dict(),"../TecXLM.pth")
model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more_generated.txt', 'w').write(decode(m0del.generate(context, max_new_tokens=10000)[0].tolist()))
open('TecXLM_generating.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))

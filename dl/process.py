#retrieve the model
import torch
import torch.nn as nn
from model import UNetGenerator

model = UNetGenerator()
model.load_state_dict(torch.load("./model.pt"))
model.eval()
# utils.py
import torch
def device_str():
    return "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = torch.device("cuda")
model.to(device)
activations = {}
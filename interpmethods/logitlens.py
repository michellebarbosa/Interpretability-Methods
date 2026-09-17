import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "gpt2" 
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

device = torch.device("cpu")
model.to(device)
activations = {}

# Dynamically fetch the number of layers (12 for GPT-2)
num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layer"))
layer_list = range(1, num_layers + 1)

def get_hook(layer_num):
    def hook(model, input, output):
        # FIX: Check if the output is a tuple. 
        # If it's already a tensor, taking output[0] strips the batch dimension and causes the 2D error.
        hidden_states = output[0] if isinstance(output, tuple) else output
        activations[layer_num] = hidden_states.detach()
    return hook

def register_hooks():
    list_of_hooks = []
    for i in layer_list:
        list_of_hooks.append(model.transformer.h[i-1].register_forward_hook(get_hook(i)))
    return list_of_hooks

hooks = register_hooks()

prompt = "Trump works at McDonald's. Trump works at"
tokenizer.pad_token = tokenizer.eos_token 

input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(device)

with torch.no_grad():
    fwd_pass = model(input_ids)

layer, token_pos = 2, -1

# Now activations[layer] is guaranteed to be 3D: [batch_size, seq_len, hidden_dim]
normalized_act = model.transformer.ln_f(activations[layer][0, token_pos, :])
probabilities = F.softmax(model.lm_head(normalized_act), dim=0)

max_index = torch.argmax(probabilities).item()
top_token = tokenizer.decode([max_index])

print(f"Layer {layer} predicts: '{top_token}'")

# Clean up hooks so they don't leak memory on future forward passes
for h in hooks:
    h.remove()
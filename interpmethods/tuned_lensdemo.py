import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuned_lens import TunedLens

MODEL_ID = "gpt2"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 1. The Correct Load Method: Fetches the pre-trained affine translators for GPT-2
tuned_lens = TunedLens.from_model_and_pretrained(model, map_location="cpu")

prompt = "Trump works at McDonald's. Trump works at"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

print("Tuned Lens Predictions across layers:\n")

with torch.no_grad():
    # 2. Tell the model to hand over its intermediate thoughts automatically!
    # No custom hooks required. It returns a tuple of all layer activations.
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states 
    
    # 3. Apply the Tuned Lens to each intermediate layer's hidden state
    for layer in range(len(hidden_states) - 1):
        h = hidden_states[layer]
        
        # THE MAGIC: This applies the Affine Translation AND the final output matrix
        logits = tuned_lens.forward(h, layer)
        
        # Look at the probabilities for the very last token in the prompt
        probabilities = F.softmax(logits[0, -1, :], dim=-1)
        max_index = torch.argmax(probabilities).item()
        top_token = tokenizer.decode([max_index])
        
        print(f"Layer {layer:2d} translated prediction: '{top_token}'")
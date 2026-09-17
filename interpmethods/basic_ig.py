import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients

# 1. Load a pre-trained sentiment model and its tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 2. Write a tiny wrapper function for Captum
# Captum needs a function that takes inputs and returns JUST the prediction numbers (logits)
def predict_forward(input_ids):
    # Pass the inputs to the model and grab the raw prediction scores
    return model(input_ids).logits

# 3. Hook Captum into the model's embedding layer
# For DistilBERT, the word embeddings live here:
embedding_layer = model.distilbert.embeddings.word_embeddings
lig = LayerIntegratedGradients(predict_forward, embedding_layer)

# 4. Prepare the text
sentence = "The dog was absolutely wild."
# Tokenize the sentence (adds [CLS] at the start and [SEP] at the end)
tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
input_ids = tokenizer.encode(sentence, return_tensors="pt") # Shape: [1, seq_length]

# 5. Create the Baseline (a sequence of [PAD] tokens of the exact same length)
pad_token_id = tokenizer.pad_token_id
baseline_ids = torch.tensor([[pad_token_id] * input_ids.shape[1]])

# 6. Run Integrated Gradients!
# target=1 means "Class 1" (Positive Sentiment in this model)
attributions, delta = lig.attribute(
    inputs=input_ids,
    baselines=baseline_ids,
    target=1, 
    return_convergence_delta=True
)

# 7. Sum across the embedding dimensions to get one score per token
word_importances = attributions.sum(dim=-1).squeeze(0) # Remove batch dimension

# 8. Print the results nicely
print(f"Analyzing Sentence: '{sentence}'\n")
print(f"{'Token':<15} | {'Score':>7} | Impact on POSITIVE prediction")
print("-" * 55)

for token, score in zip(tokens, word_importances.tolist()):
    impact = "🟢 Pushed toward Positive" if score > 0 else "🔴 Pushed toward Negative"
    
    # Optional: Fade out near-zero scores to highlight important words
    if abs(score) < 0.15:
        impact = "⚪ Neutral"
        
    print(f"{token:<15} | {score:>7.3f} | {impact}")

print("-" * 55)
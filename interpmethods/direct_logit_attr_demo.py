import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")

prompt = "The woman worked as a"

target_token = " nurse"
distractor_token = " doctor"

target_id = model.to_single_token(target_token)
distractor_id = model.to_single_token(distractor_token)
logit_diff_dir = model.W_U[:, target_id] - model.W_U[:, distractor_id]

def get_head_dla(prompt):
    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)
    per_head_residual = cache.stack_head_results(return_labels=False, apply_ln=True)
    last_token_residual = per_head_residual[:, :, -1, :]
    return (last_token_residual * logit_diff_dir).sum(dim=-1)

dla = get_head_dla(prompt)

save_dir = "/Users/michelle/Desktop/Interpretability-Methods/visualizations"
os.makedirs(save_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    dla.detach().cpu().numpy(),
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".3f",
    cbar_kws={'label': 'DLA'}
)
plt.title(f"DLA — \"{prompt}\"\nTarget: '{target_token}' vs '{distractor_token}'")
plt.xlabel("Attention Head")
plt.ylabel("Layer")
plt.gca().invert_yaxis()

plt.savefig(os.path.join(save_dir, "dla_dalit.png"), dpi=300, bbox_inches="tight")
plt.close()
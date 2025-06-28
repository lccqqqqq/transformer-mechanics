import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import load_model_and_tokenizer
import os
import pickle
os.chdir("/mnt/users/clin/workspace/mathematics_transformer")
os.makedirs("data/embedding", exist_ok=True)
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import time
import numpy as np
import einops


### ----- Get embedding and unembedding matrices -----

model, tokenizer = load_model_and_tokenizer("gpt2-small", format="nns")
embedding_matrix = model.get_input_embeddings().weight.detach().cuda()  # Move to GPU
unembedding_matrix = model.get_output_embeddings().weight.detach().cuda()  # Move to GPU

### ----- Similarity between embedding and unembedding matrices -----

# Calculate cosine similarity between corresponding rows
cosine_similarities = []
norm_ratios = []
eps = 1e-8
for i in tqdm(range(embedding_matrix.shape[0])):
    emb_vec = embedding_matrix[i]
    unemb_vec = unembedding_matrix[i]
    
    # Calculate cosine similarity
    dot_product = t.dot(emb_vec, unemb_vec)
    norm_emb = t.norm(emb_vec)
    norm_unemb = t.norm(unemb_vec)
    cosine_sim = dot_product / (norm_emb * norm_unemb + eps)
    cosine_similarities.append(cosine_sim.item())
    
    # get norm ratio
    norm_ratio = norm_emb / (norm_unemb + eps)
    norm_ratios.append(norm_ratio.item())

# save cosine similarities and norm ratios
with open("data/embedding/embed_unembed_cossim.pkl", "wb") as f:
    pickle.dump(cosine_similarities, f)
with open("data/embedding/embed_unembed_normratios.pkl", "wb") as f:
    pickle.dump(norm_ratios, f)
    
print(f"Saved {len(cosine_similarities)} cosine similarities to data/embedding/embed_unembed_cossim.pkl")
print(f"Saved {len(norm_ratios)} norm ratios to data/embedding/embed_unembed_normratios.pkl")


### ----- Verifying orthogonality -----

t.manual_seed(0)
trunc = 50000
rand_index_set = t.randint(0, embedding_matrix.shape[0], size=(trunc, 2)).cuda()  # Move to GPU
# Mask out cases where we have identical indices (i.e., same token for embedding and unembedding)
mask = rand_index_set[:, 0] != rand_index_set[:, 1]
rand_index_set = rand_index_set[mask]  # Shape: [trunc, 2]

# Get the selected vectors
rand_embed_vecs = embedding_matrix[rand_index_set[:, 0]]
rand_unembed_vecs = embedding_matrix[rand_index_set[:, 1]]
cosine_similarities = t.nn.functional.cosine_similarity(rand_embed_vecs, rand_unembed_vecs, dim=1)

# save random indices, cosine similarities as tensor
data_dict = {
    'rand_indices': rand_index_set.to("cpu"),
    'cosine_similarities': cosine_similarities.to("cpu").to(t.float32),
}
t.save(data_dict, "data/embedding/encoder_orthogonality.pth")

print(f"Saved {len(cosine_similarities)} cosine similarities with random indices to data/embedding/encoder_orthogonality.pth")

# complement with a random baseline
# This is supposed to sample uniformly randomly the directions on the unit sphere
random_matrix = t.randn(embedding_matrix.shape[0], embedding_matrix.shape[1]).cuda()
rand_vecs_1 = random_matrix[rand_index_set[:, 0]]
rand_vecs_2 = random_matrix[rand_index_set[:, 1]]
baseline_cosine_similarities = t.nn.functional.cosine_similarity(rand_vecs_1, rand_vecs_2, dim=1)

# save random indices, cosine similarities as tensor
rand_baseline_data_dict = {
    'rand_indices': rand_index_set.to("cpu"),
    'cosine_similarities': baseline_cosine_similarities.to("cpu").to(t.float32),
}
t.save(rand_baseline_data_dict, "data/embedding/encoder_orthogonality_baseline.pth")
print(f"Saved {len(baseline_cosine_similarities)} cosine similarities with random indices to data/embedding/encoder_orthogonality_baseline.pth")




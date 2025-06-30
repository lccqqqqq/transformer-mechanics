# Analyzing the thinking process of next-token predictors
# basically using logit lens
# The hypothesis: the residue stream starts with something aligned with a token. Then at intermediate steps it starts to deviate from specific tokens, i.e. the inverse participation ratio starts to increase and the probability becomes more uniform across tokens. (careful to use logits/probabilities, maybe the logit is a more standardized choice). At late stages, the residue stream is supposed to come back to the tokens.

import torch as t
from nnsight import LanguageModel
from utils import load_model_and_tokenizer
from jaxtyping import Float
from torch import Tensor
from typing import List
import json

model_name = "gpt2-small"
device = "cuda"
with open("prompt_list.json", "r") as f:
    prompt_list = json.load(f)

def get_probs_all_layers(model_name: str, device: str, prompt: str):
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    model.eval()

    # W_E = model.get_input_embeddings().weight.detach().to(device)
    # W_U = model.get_output_embeddings().weight.detach().to(device)
    probs_layers = []
    
    with t.no_grad():
        with model.trace() as tracer:
            # the invoker actually computes the graph
            with tracer.invoke(prompt) as invoker:
                for layer_idx, layer in enumerate(model.transformer.h):
                    layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                    probs = t.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)
            
        probs_all_layers = t.cat([probs.value for probs in probs_layers])
    
    return probs_all_layers

def IPR_metric(probs_all_layers: Float[Tensor, "layers vocab"], alpha: float = 2.0):
    # inverse participation ratio along the layers
    # IPR = 1 / sum(p_i^2) where p_i are the probabilities
    # This measures the localization of the distribution
    
    # Calculate sum of squared probabilities for each layer
    sum_squared_probs = t.sum(probs_all_layers ** alpha, dim=-1)
    
    # Calculate inverse participation ratio
    ipr = 1.0 / sum_squared_probs
    
    return ipr

def entropy_metric(probs_all_layers: Float[Tensor, "layers vocab"]):
    # entropy of the distribution
    # H = -sum(p_i * log(p_i))
    # This measures how "uniform" the distribution is
    entropy = -t.sum(probs_all_layers * t.log(probs_all_layers), dim=-1)
    return entropy

def top_k_metric(probs_all_layers: Float[Tensor, "layers vocab"], k: int = 10):
    # Using top-k to measure how much the distribution is concentrated on the top-k tokens
    values, indices = t.topk(probs_all_layers, k=k, dim=-1)
    return (values).sum(dim=-1)

def main(
    model_name: str = "gpt2-small",
    device: str = "cuda",
    prompt_label: str = "gpt4o-normal",
    k: int = 10,
    alpha: float = 2.0,
    save_dir: str = "data/thinking"
):
    prompt = prompt_list[prompt_label]
    probs_all_layers = get_probs_all_layers(model_name, device, prompt)
    ipr = IPR_metric(probs_all_layers)
    entropy = entropy_metric(probs_all_layers)
    top_k = top_k_metric(probs_all_layers, k=k)

    # save the results
    results = {
        "ipr": ipr,
        "entropy": entropy,
        "top_k": top_k,
    }
    t.save(results, f"{save_dir}/results_{model_name}_{prompt_label}.pt")

def debug():
    from utils import print_shape
    model_name = "gpt2-small"
    device = "cuda"
    prompt = "The cat sat on the mat."
    # probs_all_layers = get_probs_all_layers(model_name, device, prompt)
    # print_shape(probs_all_layers)
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    print(tokenizer.encode(prompt, add_special_tokens=False))
    print([tokenizer.decode(tokenizer.encode(prompt, add_special_tokens=False)[i]) for i in range(len(tokenizer.encode(prompt, add_special_tokens=False)))])
    


if __name__ == "__main__":
    main(model_name="gpt2-xl", prompt_label="gpt2xl-indist")
    # debug()

    

    
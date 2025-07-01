# Analyzing the thinking process of next-token predictors
# basically using logit lens
# The hypothesis: the residue stream starts with something aligned with a token. Then at intermediate steps it starts to deviate from specific tokens, i.e. the inverse participation ratio starts to increase and the probability becomes more uniform across tokens. (careful to use logits/probabilities, maybe the logit is a more standardized choice). At late stages, the residue stream is supposed to come back to the tokens.

import torch as t
from nnsight import LanguageModel
from utils import load_model_and_tokenizer
from jaxtyping import Float
from torch import Tensor
from jaxtyping import Int
from typing import Callable
import json

model_name = "gpt2-small"
device = "cuda"
with open("prompt_list.json", "r") as f:
    prompt_list = json.load(f)

def get_probs_all_layers(model_name: str, device: str, prompt: str):
    """
    Extract probability distributions from all transformer layers for a given prompt.
    
    This function performs a forward pass through the model and captures the probability
    distributions at each layer's output, effectively implementing a "logit lens" approach
    to analyze how the model's internal representations evolve across layers.
    
    Args:
        model_name (str): Name of the model to load (e.g., "gpt2-small", "gpt2-xl")
        device (str): Device to run the model on ("cuda" or "cpu")
        prompt (str): Input text prompt to analyze
        
    Returns:
        Float[Tensor, "layers seq_len vocab"]: Probability distributions for each layer,
            sequence position, and vocabulary token. Shape is [num_layers, seq_len, vocab_size].
            
    Example:
        >>> probs = get_probs_all_layers("gpt2-small", "cuda", "The cat sat on the mat.")
        >>> print(probs.shape)  # [12, 6, 50257] for GPT-2 small
    """
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    model.eval()

    # W_E = model.get_input_embeddings().weight.detach().to(device)
    # W_U = model.get_output_embeddings().weight.detach().to(device)
    probs_layers = []
    
    with t.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                layer_0_input = model.transformer.h[0].input
                # shape should be [ctx d_model]
                # The input to the first layer, corresponds to the initial token embeddings plus position embeddings
                
                probs = t.nn.functional.softmax(model.lm_head(model.transformer.ln_f(layer_0_input)), dim=-1).save()
                probs_layers.append(probs)
                
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

def IPR_metric(probs_all_layers: Float[Tensor, "... vocab"], alpha: float = 2.0):
    """
    Calculate the Inverse Participation Ratio (IPR) of the decoded residue stream across layers.
    
    The IPR measures how localized or concentrated a probability distribution is.
    A low IPR indicates the distribution is concentrated on few tokens (localized),
    while a high IPR indicates a more uniform distribution across many tokens.
    
    IPR = 1 / sum(p_i^alpha) where p_i are the probabilities and alpha is a parameter.
    For alpha=2, this is the standard inverse participation ratio.
    
    Args:
        probs_all_layers (Float[Tensor, "... vocab"]): Probability distributions across
            layers, sequence positions, and vocabulary tokens
        alpha (float, optional): Exponent for the IPR calculation. Defaults to 2.0.
            Higher values make the metric more sensitive to concentration.
            
    Returns:
        Float[Tensor, "..."]: Inverse participation ratio values with the same shape
            as input except for the vocabulary dimension.
            
    Example:
        >>> probs = get_probs_all_layers("gpt2-small", "cuda", "Hello world")
        >>> ipr = IPR_metric(probs, alpha=2.0)
        >>> print(ipr.shape)  # [12, 2] for 12 layers, 2 tokens
    """
    # inverse participation ratio along the layers
    # IPR = 1 / sum(p_i^2) where p_i are the probabilities
    # This measures the localization of the distribution
    
    # Calculate sum of squared probabilities for each layer
    sum_squared_probs = t.sum(probs_all_layers ** alpha, dim=-1)
    
    # Calculate inverse participation ratio
    ipr = 1.0 / sum_squared_probs
    
    return ipr

def entropy_metric(probs_all_layers: Float[Tensor, "... vocab"]):
    """
    Calculate the entropy of probability distributions, decoded from the residue stream, across layers.
    
    Entropy measures the uncertainty or uniformity of a probability distribution.
    Higher entropy indicates a more uniform distribution across tokens,
    while lower entropy indicates concentration on fewer tokens.
    
    H = -sum(p_i * log(p_i)) where p_i are the probabilities.
    
    Args:
        probs_all_layers (Float[Tensor, "... vocab"]): Probability distributions across
            layers, sequence positions, and vocabulary tokens
            
    Returns:
        Float[Tensor, "..."]: Entropy values with the same shape as input except
            for the vocabulary dimension.
            
    Example:
        >>> probs = get_probs_all_layers("gpt2-small", "cuda", "Hello world")
        >>> entropy = entropy_metric(probs)
        >>> print(entropy.shape)  # [12, 2] for 12 layers, 2 tokens
    """
    # entropy of the distribution
    # H = -sum(p_i * log(p_i))
    # This measures how "uniform" the distribution is
    entropy = -t.sum(probs_all_layers * t.log(probs_all_layers), dim=-1)
    return entropy

def top_k_metric(probs_all_layers: Float[Tensor, "... vocab"], k: int = 10):
    """
    Calculate the sum of top-k probabilities, decoded from the residue stream, across layers.
    
    This metric measures how much probability mass is concentrated on the top-k
    most likely tokens. Higher values indicate the distribution is more concentrated
    on a small set of tokens, while lower values indicate more uniform distribution.
    
    Args:
        probs_all_layers (Float[Tensor, "... vocab"]): Probability distributions across
            layers, sequence positions, and vocabulary tokens
        k (int, optional): Number of top probabilities to sum. Defaults to 10.
            
    Returns:
        Float[Tensor, "..."]: Sum of top-k probabilities with the same shape as input
            except for the vocabulary dimension.
            
    Example:
        >>> probs = get_probs_all_layers("gpt2-small", "cuda", "Hello world")
        >>> top_k_sum = top_k_metric(probs, k=5)
        >>> print(top_k_sum.shape)  # [12, 2] for 12 layers, 2 tokens
    """
    # Using top-k to measure how much the distribution is concentrated on the top-k tokens
    values, indices = t.topk(probs_all_layers, k=k, dim=-1)
    return values.sum(dim=-1)

def main(
    model_name: str = "gpt2-small",
    device: str = "cuda",
    prompt_label: str = "gpt4o-normal",
    k: int = 10,
    alpha: float = 2.0,
    save_dir: str = "data/thinking"
):
    """
    Main function to analyze thinking process metrics and save results.
    
    This function orchestrates the complete analysis pipeline: extracting probability
    distributions from all layers, computing various metrics (IPR, entropy, top-k),
    and saving the results to disk for further analysis.
    
    Args:
        model_name (str, optional): Name of the model to analyze. Defaults to "gpt2-small".
        device (str, optional): Device to run the model on. Defaults to "cuda".
        prompt_label (str, optional): Key to access prompt from prompt_list.json.
            Defaults to "gpt4o-normal".
        k (int, optional): Number of top probabilities for top_k_metric. Defaults to 10.
        alpha (float, optional): Exponent for IPR calculation. Defaults to 2.0.
        save_dir (str, optional): Directory to save results. Defaults to "data/thinking".
            
    Returns:
        None: Results are saved to disk as a PyTorch tensor file.
        
    Example:
        >>> main(model_name="gpt2-xl", prompt_label="math-problem", k=5)
        # Saves results to data/thinking/results_gpt2-xl_math-problem.pt
    """
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
    t.save(results, f"{save_dir}/results_{model_name}_{prompt_label}_top{k}.pt")
    
def context_dependent_weight(seq_len: int, position: Int[Tensor, ""], beta: float = 1.0):
    """
    Calculate context-dependent weights for sequence positions.
    
    This function computes weights that increase exponentially with the amount of
    context available at each position. Positions later in the sequence have
    access to more context and thus receive higher weights.
    
    Args:
        seq_len (int): Total length of the sequence
        position (Int[Tensor, ""]): Current position in the sequence (0-indexed)
        beta (float, optional): Scaling parameter for the exponential weight.
            Defaults to 1.0. Higher values create steeper weight gradients.
            
    Returns:
        Float[Tensor, ""]: Context-dependent weight for the given position.
            Weight increases exponentially from position 0 to seq_len-1.
            
    Example:
        >>> weight = context_dependent_weight(10, t.tensor(5), beta=1.0)
        >>> print(weight)  # Higher weight for position 5 in sequence of length 10
    """
    # Weight based on how much context the model has seen
    context_ratio = position / seq_len  # assuming shape [layers, seq_len, vocab]
    return t.exp(beta * (context_ratio - 1))  # Exponential increase with context

def context_dependent_weight_total(seq_len: int):
    """
    Calculate the total sum of context-dependent weights for all positions.
    
    This function computes the normalization factor for context-dependent weights
    across an entire sequence, useful for normalizing weighted averages.
    
    Args:
        seq_len (int): Total length of the sequence
        
    Returns:
        float: Sum of all context-dependent weights from position 0 to seq_len-1.
            
    Example:
        >>> total_weight = context_dependent_weight_total(10)
        >>> print(total_weight)  # Sum of weights for positions 0-9
    """
    all_context_ratios = t.arange(seq_len) / seq_len
    return t.exp(all_context_ratios - 1).sum()

def debug():
    from utils import print_shape
    model_name = "gpt2-xl"
    device = "cuda"
    prompt = "The cat sat on the mat."
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    normal_output, no_pos_output = exclude_position_emb(model, prompt)
    print_shape(normal_output)
    print_shape(no_pos_output)
    
    print("--------------------------------")
    normal_metric, no_pos_metric = compare_closeness_to_tokens_without_pos_emb(model_name, prompt_label="gpt2xl-indist", metric=top_k_metric)
    print_shape(normal_metric)
    print_shape(no_pos_metric)


def exclude_position_emb(model: LanguageModel, prompt: str):
    """
    Compare model outputs with and without position embeddings.
    
    This function performs two forward passes: one with normal position embeddings
    and another with zeroed position embeddings. It returns the probability
    distributions from the first layer for both cases, allowing analysis of
    how position embeddings affect the model's internal representations.
    
    Args:
        model (LanguageModel): Loaded language model instance
        prompt (str): Input text prompt to analyze
        
    Returns:
        tuple: Two tensors containing probability distributions:
            - resid_probs: Normal forward pass probabilities
            - resid_probs_no_pos: Forward pass with zeroed position embeddings
            Both have shape [seq_len, vocab_size].
            
    Example:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2-small", "cuda")
        >>> normal_probs, no_pos_probs = exclude_position_emb(model, "Hello world")
        >>> print(normal_probs.shape, no_pos_probs.shape)  # [2, 50257], [2, 50257]
    """
    # Compare with and without position embeddings
    
    with t.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                # Normal forward pass
                normal_output = model.transformer.h[0].output[0][0].save()
                # shape [seq_len, hidden_size]
        
        model.transformer.wpe.weight.data.fill_(0)    
        with model.trace() as tracer:
            with tracer.invoke(prompt) as invoker:
                # Zero out position embeddings            
                no_pos_output = model.transformer.h[0].output[0][0].save()
        
    # decode the resid output
    resid_output = model.lm_head(model.transformer.ln_f(normal_output))
    resid_probs = t.nn.functional.softmax(resid_output, dim=-1)
    
    resid_output_no_pos = model.lm_head(model.transformer.ln_f(no_pos_output))
    resid_probs_no_pos = t.nn.functional.softmax(resid_output_no_pos, dim=-1)
    
    return resid_probs, resid_probs_no_pos

def compare_closeness_to_tokens_without_pos_emb(
    model_name: str,
    prompt_label: str,
    metric: Callable,
):
    """
    Compare metric values with and without position embeddings.
    
    This function analyzes how position embeddings affect various metrics
    (IPR, entropy, top-k) by comparing the metric values computed on outputs
    with normal position embeddings versus zeroed position embeddings.
    
    Args:
        model_name (str): Name of the model to analyze
        prompt_label (str): Key to access prompt from prompt_list.json
        metric (Callable): Function to compute the metric (e.g., top_k_metric, IPR_metric)
        
    Returns:
        tuple: Two tensors containing metric values:
            - normal_metric: Metric computed with normal position embeddings
            - no_pos_metric: Metric computed with zeroed position embeddings
            Both have the same shape as the metric function output.
            
    Example:
        >>> normal, no_pos = compare_closeness_to_tokens_without_pos_emb(
        ...     "gpt2-xl", "math-problem", top_k_metric)
        >>> print(normal.shape, no_pos.shape)  # [1, seq_len], [1, seq_len]
    """
    model, tokenizer = load_model_and_tokenizer(model_name, device=device)
    prompt = prompt_list[prompt_label]
    resid_probs, resid_probs_no_pos = exclude_position_emb(model, prompt)
    normal_metric = metric(resid_probs)
    no_pos_metric = metric(resid_probs_no_pos)
    
    data = {
        "normal_metric": normal_metric,
        "no_pos_metric": no_pos_metric,
    }
    t.save(data, f"data/thinking/posembed_layer0_comparison_{model_name}_{prompt_label}.pt")
    
    return normal_metric, no_pos_metric
    

if __name__ == "__main__":
    main(model_name="gpt2-xl", prompt_label="gpt2xl-random", k=5)
    # debug()
    # compare_closeness_to_tokens_without_pos_emb(model_name="gpt2-xl", prompt_label="gpt4o-normal", metric=top_k_metric)


    

    
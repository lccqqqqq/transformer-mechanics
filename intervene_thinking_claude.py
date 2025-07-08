import torch as t
from nnsight import LanguageModel
from utils import load_model_and_tokenizer
from jaxtyping import Float
from torch import Tensor
from jaxtyping import Int
from typing import Callable, List, Dict, Tuple
import json
from itertools import product
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np


def interpolation_steering(
    target_vector: Float[Tensor, "... d_model"],
    original_vector: Float[Tensor, "... d_model"],
    steering_strength: float = 0.1,
):
    """
    Note:
        This method provides more controlled steering compared to direct_steering
        because the result is always a convex combination of the input vectors
        when steering_strength ≤ 1. This prevents the output from having
        unexpectedly large magnitudes.
    """
    if steering_strength > 1:
        raise ValueError("Steering strength must be less than 1")
    
    new_vector = original_vector + steering_strength * (target_vector - original_vector)
    return new_vector

def random_steering(
    target_vector: Float[Tensor, "... d_model"],
    original_vector: Float[Tensor, "... d_model"],
    steering_strength: float = 0.1,
):
    """
    Note:
        This method steers the original vector towards a random direction.
    """
    random_vector = t.randn_like(original_vector)
    new_vector = original_vector + steering_strength * random_vector / t.norm(random_vector)
    return new_vector


def batch_run_steering(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    steering_strength: float = 0.1,
    steering_method: Callable = None,
    max_length: int = None,
    batch_size: int = 8,
):
    """
    Batch version of run_steering that processes multiple prompts efficiently
    """
    model.eval()
    decoder = model.lm_head.weight.data
    
    if max_length is None:
        max_length = model.config.n_positions
    
    # Tokenize all prompts with padding
    tokenized = tokenizer(
        prompts, 
        max_length=max_length, 
        truncation=True, 
        padding=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    n_prompts = len(prompts)
    n_layers = len(model.transformer.h)
    batch_intervention_outputs = []
    
    with t.no_grad():
        with model.trace() as tracer:
            # Process in batches to manage memory
            for batch_start in tqdm(range(0, n_prompts, batch_size), desc="Processing batches"):
                batch_end = min(batch_start + batch_size, n_prompts)
                batch_input_ids = input_ids[batch_start:batch_end]
                batch_attention_mask = attention_mask[batch_start:batch_end]
                
                batch_outputs = []
                
                # First, get clean run for this batch
                with tracer.invoke(batch_input_ids, attention_mask=batch_attention_mask) as invoker:
                    original_output = t.nn.functional.softmax(model.lm_head.output, dim=-1).save()
                
                # Then run interventions for each layer
                for layer_idx, layer in enumerate(tqdm(model.transformer.h, desc=f"Layers (batch {batch_start//batch_size + 1})", leave=False)):
                    with tracer.invoke(batch_input_ids, attention_mask=batch_attention_mask) as invoker:
                        # Get layer output for the batch
                        layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                        layer_output_tokens = t.argmax(layer_output, dim=-1)
                        
                        # Apply steering to each item in the batch
                        if steering_method is not None:
                            target_vectors = decoder[layer_output_tokens]
                            layer.output[0] = steering_method(
                                target_vectors, 
                                layer.output[0], 
                                steering_strength
                            )
                        
                        # Get steered output
                        steered_output = t.nn.functional.softmax(model.lm_head.output, dim=-1).save()
                        batch_outputs.append(steered_output)
                
                # Combine original + all layer interventions
                batch_all_outputs = [original_output] + batch_outputs
                batch_intervention_outputs.append(t.stack(batch_all_outputs))
    
    # Concatenate all batches: shape (n_interventions, total_prompts, seq_len, vocab_size)
    all_intervention_outputs = t.cat(batch_intervention_outputs, dim=1)
    
    return all_intervention_outputs

def batch_degradation_metric(
    prompts: List[str], 
    intervention_outputs: Float[Tensor, "n_intervention n_prompts ctx d_vocab"], 
    tokenizer: AutoTokenizer, 
    k: int = 10, 
    device: str = "cuda"
) -> Float[Tensor, "n_intervention n_prompts"]:
    """
    Compute degradation metric for batched prompts
    """
    # Tokenize ground truth for all prompts
    tokenized_prompts = tokenizer(
        prompts, 
        max_length=intervention_outputs.shape[2] + 1,  # +1 because we exclude first token
        truncation=True, 
        padding=True,
        return_tensors="pt"
    )
    
    # Ground truth tokens (excluding first token for each prompt)
    ground_truth_tokens = tokenized_prompts["input_ids"][:, 1:].unsqueeze(0).unsqueeze(-1).to(device)
    attention_mask = tokenized_prompts["attention_mask"][:, 1:].to(device)
    
    # Get top-k predictions: shape (n_intervention, n_prompts, ctx, k)
    values, indices = t.topk(intervention_outputs[:, :, :-1, :], k=k, dim=-1)
    
    # Find matching probabilities
    matching_prob_within_topk = t.where(
        indices == ground_truth_tokens,
        values,
        t.zeros_like(values),
    )
    
    # Sum along the top-k dimension: shape (n_intervention, n_prompts, ctx)
    matching_prob_within_topk = matching_prob_within_topk.sum(dim=-1)
    
    # Apply attention mask to ignore padding tokens
    matching_prob_within_topk = matching_prob_within_topk * attention_mask.unsqueeze(0)
    
    # Average over sequence length (only considering non-padded tokens)
    seq_lengths = attention_mask.sum(dim=-1, keepdim=True).unsqueeze(0)
    matching_prob_within_topk = matching_prob_within_topk.sum(dim=-1) / seq_lengths.squeeze(-1)
    
    return matching_prob_within_topk

def efficient_multi_prompt_steering(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    steering_methods: Dict[Callable, t.Tensor],  # method -> strength values
    k: int = 10,
    batch_size: int = 8,
    save_individual: bool = False,
):
    """
    Efficiently run steering experiments across multiple prompts and methods
    """
    device = next(model.parameters()).device
    all_results = {}
    
    for steering_method, steering_strengths in steering_methods.items():
        print(f"Running {steering_method.__name__}")
        method_results = []
        
        for strength in tqdm(steering_strengths, desc="Steering strengths"):
            # Run batch steering for this strength
            intervention_outputs = batch_run_steering(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                steering_strength=float(strength),
                steering_method=steering_method,
                batch_size=batch_size,
            )
            
            # Compute degradation metric for all prompts
            degradation_scores = batch_degradation_metric(
                prompts=prompts,
                intervention_outputs=intervention_outputs,
                tokenizer=tokenizer,
                k=k,
                device=device
            )
            
            method_results.append(degradation_scores)
            
            # Optionally save individual results
            if save_individual:
                filename = f"data/thinking/batch_{steering_method.__name__}_{len(prompts)}prompts_strength{strength:.2f}.pt"
                t.save({
                    'intervention_outputs': intervention_outputs,
                    'degradation_scores': degradation_scores,
                    'prompts': prompts,
                    'strength': strength
                }, filename)
        
        # Stack results across strengths: shape (n_strengths, n_interventions, n_prompts)
        all_results[steering_method] = t.stack(method_results)
    
    return all_results

def load_and_prepare_prompts(
    dataset_name: str = "stas/openwebtext-10k",
    num_prompts: int = 100,
    max_length: int = 512,
    min_length: int = 50,
    seed: int = 42
) -> List[str]:
    """
    Load and prepare prompts from dataset
    """
    ds = load_dataset(dataset_name)
    ds = ds.shuffle(seed=seed)
    
    prompts = []
    for i in range(len(ds['train'])):
        text = ds['train'][i]['text'].strip()
        
        # Filter by length (rough character count)
        if min_length <= len(text) <= max_length * 4:  # Rough char to token ratio
            prompts.append(text)
            
        if len(prompts) >= num_prompts:
            break
    
    return prompts[:num_prompts]

def main_batch_efficient():
    """
    Main function for efficient batch processing
    """
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.float32)
    
    # Load prompts
    prompts = load_and_prepare_prompts(
        num_prompts=50,  # Start smaller for testing
        max_length=256,  # Shorter sequences for memory efficiency
    )
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Define steering methods and strengths
    n_steering_strength = 5  # Reduced for efficiency
    steering_methods = {
        interpolation_steering: t.linspace(0, 1, n_steering_strength),
        random_steering: t.linspace(0, 1, n_steering_strength),
    }
    
    # Run experiments
    results = efficient_multi_prompt_steering(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        steering_methods=steering_methods,
        k=10,
        batch_size=4,  # Adjust based on GPU memory
        save_individual=True
    )
    
    # Save final aggregated results
    t.save({
        'results': results,
        'prompts': prompts,
        'steering_methods': list(steering_methods.keys()),
        'steering_strengths': {method.__name__: strengths for method, strengths in steering_methods.items()}
    }, "data/thinking/batch_results_aggregated.pt")
    
    return results

def analyze_batch_results(results_path: str = "data/thinking/batch_results_aggregated.pt"):
    """
    Analyze and visualize batch results
    """
    data = t.load(results_path)
    results = data['results']
    prompts = data['prompts']
    
    print(f"Results for {len(prompts)} prompts:")
    
    for method_name, method_results in results.items():
        print(f"\n{method_name.__name__}:")
        # method_results shape: (n_strengths, n_interventions, n_prompts)
        
        # Average across prompts for each strength and intervention
        avg_across_prompts = method_results.mean(dim=-1)  # (n_strengths, n_interventions)
        
        print(f"  Shape: {avg_across_prompts.shape}")
        print(f"  Mean degradation by layer (averaged across prompts and strengths):")
        print(f"    Min: {avg_across_prompts.min():.4f}")
        print(f"    Max: {avg_across_prompts.max():.4f}")
        print(f"    Mean: {avg_across_prompts.mean():.4f}")

# Memory optimization utilities
def estimate_memory_usage(n_prompts: int, max_seq_len: int, n_layers: int, vocab_size: int = 50257):
    """
    Estimate memory usage for batch processing
    """
    # Rough estimate in GB
    bytes_per_float = 4  # float32
    intervention_outputs_size = n_prompts * (n_layers + 1) * max_seq_len * vocab_size * bytes_per_float
    gb_size = intervention_outputs_size / (1024**3)
    
    print(f"Estimated memory for intervention outputs: {gb_size:.2f} GB")
    return gb_size

def get_optimal_batch_size(available_memory_gb: float, n_prompts: int, max_seq_len: int, n_layers: int):
    """
    Suggest optimal batch size based on available memory
    """
    single_batch_gb = estimate_memory_usage(1, max_seq_len, n_layers) / n_prompts
    optimal_batch = int(available_memory_gb / single_batch_gb)
    return max(1, min(optimal_batch, n_prompts))

if __name__ == "__main__":
    # Estimate memory first
    estimate_memory_usage(n_prompts=100, max_seq_len=512, n_layers=48)  # GPT2-XL has 48 layers
    
    # Run efficient batch processing
    results = main_batch_efficient()
    
    # Analyze results
    analyze_batch_results()
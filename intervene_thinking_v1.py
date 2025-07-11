import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch as t
from nnsight import LanguageModel, util
from utils import load_model_and_tokenizer, print_shape, print_gpu_memory, MemoryMonitor, clear_memory, interpolation_steering, random_steering
from jaxtyping import Float
from torch import Tensor
from jaxtyping import Int
from typing import Callable, List, Dict, Tuple
import json
from itertools import product
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import numpy as np
import gc
from nnsight.tracing.graph import Proxy

from thinking import topk_decoder
from debug_thinking_interventions import get_probs_all_layers_with_intervention




def run_steering_one_batch(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    input_ids: Int[Tensor, "batch_size max_length"],
    steering_strength: float = 0.1,
    steering_method: Callable | None = None,
    layer_batch_size: int = 4,
    monitor: MemoryMonitor = None,
) -> Float[Tensor, "batch_size layer ctx vocab"]:
    model.eval()
    decoder = model.lm_head.weight.data
    
    
    # print_shape(batch_input_ids)
    # First off, a clean run
    output_list = []
    with t.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(input_ids) as invoker:
                original_output = t.nn.functional.softmax(model.lm_head.output, dim=-1).save()
                # print_shape(original_output)
                if monitor is not None:
                    monitor.measure("Original output")
    
    output_list.append(original_output.unsqueeze(1))
    if monitor is not None:
        monitor.measure("Original output")
    
    layer_batches = list(range(0, model.config.n_layer, layer_batch_size))  # 48 layers (0-47) batched by batch_size
    for i, layer_batch in enumerate(layer_batches):
        batch_output = []
        with t.no_grad():
            with model.trace() as tracer:
                # Then, run interventions for each layer
                for layer_idx, layer in enumerate(model.transformer.h[layer_batch:min(layer_batch+layer_batch_size, model.config.n_layer)]):
                    with tracer.invoke(input_ids) as invoker:
                        layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                        layer_output_tokens = t.argmax(layer_output, dim=-1)
                        target_vectors = decoder[layer_output_tokens]
                        
                        # apply the intervention to the hidden state
                        # Note: one must use the in place assignment
                        if steering_method is not None:
                            hidden_state = layer.output[0]
                            steered_state = steering_method(target_vectors, hidden_state, steering_strength)
                            hidden_state[:] = steered_state
                        
                        output = t.nn.functional.softmax(model.lm_head.output, dim=-1)
                        batch_output.append(output)

                batch_output = t.stack(batch_output, dim=1)
                batch_output = batch_output.save()
        
        output_list.append(batch_output)
        if monitor is not None:
            monitor.measure(f"Batched process from {layer_batch} to {min(layer_batch+layer_batch_size, model.config.n_layer-1)}")

    return t.cat(output_list, dim=1)



def debug_run_steering_one_batch():
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.bfloat16)
    prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # set the test parameters
    test_steering_layer = 39
    test_steering_strength = 0.8
    output = run_steering_one_batch(model, tokenizer, input_ids, steering_method=interpolation_steering, steering_strength=test_steering_strength)
    
    probs_all_layers = get_probs_all_layers_with_intervention(
        model,
        tokenizer,
        prompt,
        steering_method=interpolation_steering,
        steering_strength=test_steering_strength,
        steering_layer=test_steering_layer
    )
    
    print("Test 1: The steered output from the function should be the same as the final layer output from the probs_all_layers object")
    
    try:
        assert t.allclose(
            output[0][test_steering_layer+1],
            probs_all_layers[-1]
        )
        print("Test 1 passed")
    except Exception as e:
        print(e)
        print(output[0][test_steering_layer+1])
        print(probs_all_layers[-1])


def process_one_batch(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_method_list: List[Callable] = [interpolation_steering, random_steering],
    steering_strength_list: List[float] = [0.1, 0.5, 0.8, 0.99],
    binary: bool = True,
    monitor: MemoryMonitor = None,
    k: int = 3,
    save_dir: str = None,
    save_name_suffix: str = "",
    save_data: bool = True,
) -> Float[Tensor, "method strength layer_plus_1"]:
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    
    acc_metric = t.zeros(len(steering_method_list), len(steering_strength_list), model.config.n_layer + 1)
    for method_idx, steering_method in enumerate(steering_method_list):
        for strength_idx, steering_strength in enumerate(steering_strength_list):
            output = run_steering_one_batch(
                model,
                tokenizer,
                input_ids,
                steering_method=steering_method,
                steering_strength=steering_strength,
                monitor=None
            )
            accuracy = accuracy_metric(input_ids, output, tokenizer, k=k, binary=binary)
            acc_metric[method_idx, strength_idx] = accuracy
            
    if save_data:
        if save_dir is not None:
            t.save(acc_metric, os.path.join(save_dir, f"acc_metric{save_name_suffix}.pt"))
            
    return acc_metric


def process_batches(model: LanguageModel, tokenizer: AutoTokenizer, prompts: List[str], batch_size: int = 24, max_length: int = 512, steering_method: Callable = None, steering_strength: float = 0.1, monitor: MemoryMonitor = None, n_total_prompts: int = 10000, k: int = 10, save_dir: str = None):

    if max_length is None:
        max_length = model.config.n_positions
    
    # Tokenize all prompts with padding
    tokenized = tokenizer(
        prompts[:n_total_prompts], 
        max_length=max_length, 
        truncation=True, 
        padding=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    input_ids = input_ids.to(model.device)
    
    # passing the prompts to the model
    intervention_output = []
    for batch_start in tqdm(range(0, n_total_prompts, batch_size), desc="Processing batch"):
        batch_end = min(batch_start + batch_size, n_total_prompts)
        batch_input_ids = input_ids[batch_start:batch_end]
        
        output = run_steering_one_batch(
            model,
            tokenizer,
            batch_input_ids,
            steering_method=steering_method,
            steering_strength=steering_strength,
            monitor=monitor
        )
        # process the output
        # current shape [batch layer ctx vocab] = (24, 49, 512, 50257)
        accuracy = accuracy_metric(batch_input_ids, output, tokenizer, k=k)
        intervention_output.append(accuracy)
        
        clear_memory(variables_to_keep=["intervention_output", "input_ids"])
    
    intervention_output = t.cat(intervention_output, dim=0)
    if save_dir is not None:
        t.save(intervention_output, os.path.join(save_dir, f"intervention_output_{steering_method.__name__}_{int(steering_strength * 10)}e-1.pt"))
        
    return intervention_output



def sweep_steering_strength(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    batch_size: int = 16,
    max_length: int = 512,
    steering_method: Callable = interpolation_steering,
    steering_strength_list: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    n_total_prompts: int = 10000,
    k: int = 10,
    save_dir: str = None,
    save_name_suffix: str = "",
):
    intervention_output_list = []
    for steering_strength in tqdm(steering_strength_list, desc="Sweeping steering strength"):
        intervention_output = process_batches(model, tokenizer, prompts, batch_size=batch_size, max_length=max_length, steering_method=steering_method, steering_strength=steering_strength, n_total_prompts=n_total_prompts, k=k, save_dir=None)
        intervention_output_list.append(intervention_output)
        clear_memory(variables_to_keep=["intervention_output_list"])
        
    intervention_output_list = t.stack(intervention_output_list)
    if save_dir is not None:
        t.save(intervention_output_list, os.path.join(save_dir, f"intervention_output_{steering_method.__name__}{save_name_suffix}.pt"))
        
    return intervention_output_list
    
def accuracy_metric(
    input_ids: Int[Tensor, "batch_size ctx"],
    intervention_output: Float[Tensor, "batch_size layer ctx vocab"],
    tokenizer: AutoTokenizer,
    k: int = 10,
    binary: bool = True,
):
    ground_truth_tokens = input_ids[:, 1:].unsqueeze(1).unsqueeze(-1).to(intervention_output.device)
    values, indices = t.topk(intervention_output[:, :, :-1, :], k=k, dim=-1)
    # the shape: (batch_size, layer, ctx, k)
    matching_prob_within_topk = t.where(
        indices == ground_truth_tokens,
        values if not binary else t.ones_like(values),
        t.zeros_like(values),
    )
    matching_prob_within_topk = matching_prob_within_topk.sum(dim=-1).mean(dim=-1)
    return matching_prob_within_topk
        
def memory_usage_sweeps():
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.bfloat16)
    device = "cuda"
    
    ds = load_dataset("stas/openwebtext-10k")
    ds = ds.shuffle(seed=42)
    memory_usage_history = {}
    for batch_size in tqdm([8*i for i in range(1, 16)]):
        memory_usage_history[batch_size] = {}
        for max_length in [128, 256, 512, 1024]:
            try:
                clear_memory()
                monitor = MemoryMonitor()
                monitor.start()
                
                prompts = [ds['train'][i]['text'] for i in range(batch_size)]
                tokenized = tokenizer(
                    prompts, 
                    max_length=max_length, 
                    truncation=True, 
                    padding=True,
                    return_tensors="pt"
                )
                input_ids = tokenized["input_ids"]
                run_steering_one_batch(
                    model,
                    tokenizer,
                    input_ids,
                    steering_method=interpolation_steering,
                    monitor=monitor
                )
                history = monitor.report(return_history=True)
                memory_usage_history[batch_size][max_length] = history
                clear_memory()
            except Exception as e:
                memory_usage_history[batch_size][max_length] = 'oomkilled'
                print(e)
                clear_memory()
                                
    with open("memory_usage_history_141gb.json", "w") as f:
        json.dump(memory_usage_history, f)
        
    return memory_usage_history

def main():
    monitor = MemoryMonitor()
    monitor.start()
    
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.bfloat16)
    monitor.measure("Model loaded")
    device = "cuda"
    
    ds = load_dataset("stas/openwebtext-10k")
    ds = ds.shuffle(seed=42)
    prompts = [ds['train'][i]['text'] for i in range(24)]
    tokenized = tokenizer(
        prompts, 
        max_length=512, 
        truncation=True, 
        padding=True,
        return_tensors="pt"
    )
    input_ids = tokenized["input_ids"]
    output = run_steering_one_batch(model, tokenizer, input_ids, steering_method=interpolation_steering, monitor=monitor)
    monitor.measure("Steering done")
    print(output.shape)
    output.to("cpu")
    monitor.measure("Output moved to CPU")
    monitor.report()
    
def run_random_baseline_steering(trail_number: int = 5, n_total_prompts: int = 2000):
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.bfloat16)
    ds = load_dataset("stas/openwebtext-10k")
    batch_size = 16
    max_length = 512
    k = 3
    steering_method = random_steering
    save_dir = "data/thinking"
    steering_strength_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    seeds = [42 + i for i in range(trail_number)]
    for trail_idx, seed in enumerate(seeds):
        print(f"Running trail {trail_idx} with seed {seed}")
        ds = ds.shuffle(seed=seed)
        prompts = [ds['train'][i]['text'] for i in range(n_total_prompts)]
        tokenized = tokenizer(
            prompts, 
            max_length=512, 
            truncation=True, 
            padding=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"]
        
        intervention_output = sweep_steering_strength(
            model,
            tokenizer,
            prompts,
            batch_size=batch_size,
            max_length=max_length,
            steering_method=steering_method,
            steering_strength_list=steering_strength_list,
            n_total_prompts=n_total_prompts,
            k=k,
            save_dir=save_dir,
            save_name_suffix=f"_trail_{trail_idx}" if trail_idx > 1 else "_total",
        )
        
def test_process_one_batch():
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.bfloat16)
    prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    process_one_batch(model, tokenizer, prompt, save_dir="data/thinking", save_name_suffix="_test", save_data=True, binary=True)

if __name__ == "__main__":
    # n_total_prompts = 2000
    # model_name = "gpt2-xl"
    # model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.bfloat16)
    # k = 3 # using top-3 accuracy and binary accuracy metric
    
    # ds = load_dataset("stas/openwebtext-10k")
    # ds = ds.shuffle(seed=42)
    # prompts = [ds['train'][i]['text'] for i in range(n_total_prompts)]
    
    # batch_size = 16
    # max_length = 512
    # steering_strength_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # steering_method = interpolation_steering
    # save_dir = "data/thinking"
    
    # intervention_output = sweep_steering_strength(model, tokenizer, prompts, batch_size=batch_size, max_length=max_length, steering_method=steering_method, steering_strength_list=steering_strength_list, n_total_prompts=n_total_prompts, k=k, save_dir=save_dir)
    # print(f"Finished interpolation steering sweeps, starting random baseline steering sweeps")
    # print(intervention_output.shape)
    # matching_prob_within_topk = process_batches(model, tokenizer, prompts, batch_size=batch_size, max_length=max_length, steering_method=interpolation_steering, n_total_prompts=n_total_prompts, k=10)
    # print(matching_prob_within_topk.shape)
    # run_random_baseline_steering(trail_number=1, n_total_prompts=2000)
    # print(f"Finished random baseline steering sweeps")
    # debug_run_steering_one_batch()
    test_process_one_batch()
    











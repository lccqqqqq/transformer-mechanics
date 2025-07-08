from utils import load_model_and_tokenizer, print_shape, interpolation_steering, random_steering
from typing import Callable
from torch import Tensor
from jaxtyping import Float, Int
import numpy as np
from itertools import product
from tqdm import tqdm
from nnsight import LanguageModel
from transformers import AutoTokenizer
import torch as t
import json
from thinking import get_probs_all_layers, topk_decoder

# a combination of steering experiments and thinking interventions

def topk_decoder_with_intervention(
    model_name: str,
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_list: list[int],
    k: int = 3, # for visualization purposes
    steering_method: Callable | None = None,
    steering_strength: float = 0.1,
    steering_layer: int = 0,
    save_dir: str = "data/thinking",
    save_name_suffix: str = "",
    save_data: bool = True,
):
    model.eval()
    decoder = model.lm_head.weight.data
    
    probs_all_layers = get_probs_all_layers_with_intervention(
        model,
        tokenizer,
        prompt,
        steering_method,
        steering_strength,
        steering_layer
    )
    
    values, indices = t.topk(probs_all_layers[layer_list], k=k, dim=-1)
    # shapes of values: [len(layer_list), ctx, k]
    # get the topk tokens

    str_array = [[
        ["" for _ in range(k)]
        for _ in range(len(layer_list))
    ] for _ in range(probs_all_layers.shape[1])] # the context length
    
    for layer_idx, seq_pos, k_idx in tqdm(
        product(range(len(layer_list)), range(probs_all_layers.shape[1]), range(k)),
        total=len(layer_list) * probs_all_layers.shape[1] * k,
        desc="Decoding tokens"
    ):
        token_id = indices[layer_idx, seq_pos, k_idx].item()
        token_string = tokenizer.decode([token_id])
        token_string = token_string.replace(" ", "_")
        str_array[seq_pos][layer_idx][k_idx] = token_string
        
    # Save numeric data with torch.save
    numeric_data = {
        "values": values,
        "indices": indices,
    }
    t.save(numeric_data, f"{save_dir}/topk_decoder_{model_name}_top{k}{save_name_suffix}.pt")
    
    with open(f"{save_dir}/topk_decoder_{model_name}_top{k}{save_name_suffix}.json", "w") as f:
        json.dump(str_array, f, indent=2)
    
    return str_array, values, indices


def get_probs_all_layers_with_intervention(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_method: Callable,
    steering_strength: float,
    steering_layer: int,
) -> Float[Tensor, "layer_plus_1 ctx vocab"]:
    model.eval()
    decoder = model.lm_head.weight.data
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
                    if layer_idx == steering_layer and steering_method is not None:
                        layer_output = model.lm_head(
                            model.transformer.ln_f(layer.output[0])
                        )
                        layer_output_tokens = t.argmax(layer_output, dim=-1)
                        target_vectors = decoder[layer_output_tokens]
                        
                        hidden_state = layer.output[0]
                        steered_state = steering_method(
                            target_vectors,
                            hidden_state,
                            steering_strength
                        )
                        hidden_state[:] = steered_state # in place assignment of the layer output
                        
                    layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                    probs = t.nn.functional.softmax(layer_output, dim=-1).save()
                    probs_layers.append(probs)
            
        probs_all_layers = t.cat([probs.value for probs in probs_layers])
    
    return probs_all_layers



def trail():
    model_name = "gpt2-xl"
    device = "cuda"
    k = 3
    steering_strength = 0.89
    steering_layer = 46
    prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    layer_list = [47, 48]
    
    # model: LanguageModel,
    # tokenizer: AutoTokenizer,
    # prompt: str,
    # steering_method: Callable,
    # steering_strength: float,
    # steering_layer: int,
    
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    probs_all_layers_original = get_probs_all_layers_with_intervention(
        model,
        tokenizer,
        prompt,
        steering_method=None,
        steering_strength=0.0,
        steering_layer=0
    )
    print_shape(probs_all_layers_original)
    # probs_all_layers_interpolation_intervention = get_probs_all_layers_with_intervention(
    #     model,
    #     tokenizer,
    #     prompt,
    #     steering_method=interpolation_steering,
    #     steering_strength=steering_strength,
    #     steering_layer=steering_layer
    # )
    # probs_all_layers_random_intervention = get_probs_all_layers_with_intervention(
    #     model,
    #     tokenizer,
    #     prompt,
    #     steering_method=random_steering,
    #     steering_strength=steering_strength,
    #     steering_layer=steering_layer
    # )
    print("Starting clean run")
    str_array, values, indices = topk_decoder_with_intervention(
        model_name,
        model,
        tokenizer,
        prompt,
        layer_list=layer_list,
        k=k,
        steering_method=None,
        steering_strength=0.0,
        steering_layer=0,
        save_dir="data/thinking",
        save_name_suffix="_clean_run"
    )
    
    print("Starting interpolation steering")
    str_array, values, indices = topk_decoder_with_intervention(
        model_name,
        model,
        tokenizer,
        prompt,
        layer_list=layer_list,
        k=3,
        steering_method=interpolation_steering,
        steering_strength=steering_strength,
        steering_layer=steering_layer,
        save_dir="data/thinking",
        save_name_suffix="_interpolation_steering_1_layer_46_late_layer"
    )
    
    print("Starting random steering")
    str_array, values, indices = topk_decoder_with_intervention(
        model_name,
        model,
        tokenizer,
        prompt,
        layer_list=layer_list,
        k=3,
        steering_method=random_steering,
        steering_strength=steering_strength,
        steering_layer=steering_layer,
        save_dir="data/thinking",
        save_name_suffix="_random_steering_1_layer_46_late_layer"
    )
    
    # print_shape(values)
    

def debug_get_probs_all_layers_with_intervention():
    model_name = "gpt2-xl"
    device = "cuda"
    k = 3
    steering_strength = 0.2
    steering_layer = 20
    prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    layer_list = [0, 10, 21, 30, 40, 48]
    
    
    print("Test 1: With no steering, all output should be exactly the same.")
    # compare the probabilities obtained from either function
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    probs_all_layers_original = get_probs_all_layers(
        model_name,
        device,
        prompt
    )
    probs_all_layers_with_new_function = get_probs_all_layers_with_intervention(
        model,
        tokenizer,
        prompt,
        steering_method=None,
        steering_strength=0.0,
        steering_layer=0
    )
    print_shape(probs_all_layers_original)
    print_shape(probs_all_layers_with_new_function)
    
    try:
        assert t.allclose(probs_all_layers_original, probs_all_layers_with_new_function), "The probabilities are not the same"
        print("Test 1 passed")
    except AssertionError as e:
        print("Test 1 failed: ", e)
    
    # --------------------
    print("-" * 30)
    
    print("Test 2: With interpolation steering on layer L, all output before L should be the same.")
    probs_all_layers_interpolation_intervention = get_probs_all_layers_with_intervention(
        model,
        tokenizer,
        prompt,
        steering_method=interpolation_steering,
        steering_strength=steering_strength,
        steering_layer=steering_layer
    )
    print_shape(probs_all_layers_interpolation_intervention)
    try:
        assert t.allclose(probs_all_layers_original[:steering_layer+1], probs_all_layers_interpolation_intervention[:steering_layer+1]), "The probabilities are not the same"
        print("Test 2 passed")
    except AssertionError as e:
        print("Test 2 failed: ", e)

def debug_topk_decoder_with_intervention():
    model_name = "gpt2-xl"
    device = "cuda"
    k = 3
    steering_strength = 0.2
    steering_layer = 20
    prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    layer_list = [0, 10, 21, 30, 40, 48]
    
    print("Test 1: With no steering, all output should be exactly the same.")
    
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    str_array, values, indices = topk_decoder_with_intervention(
        model_name,
        model,
        tokenizer,
        prompt,
        layer_list,
        k,
        steering_method=None,
        steering_strength=0.0,
        steering_layer=0,
        save_dir="data/thinking",
        save_name_suffix="_clean_run",
        save_data=False
    )
    str_array_topk_decoder, values_topk_decoder, indices_topk_decoder = topk_decoder(
        model_name,
        layer_list,
        device,
        prompt,
        k,
        save_data=False
    )
    
    try:
        assert t.allclose(values, values_topk_decoder), "The topk decoder with intervention and the topk decoder are not the same"
        assert t.allclose(indices, indices_topk_decoder), "The topk decoder with intervention and the topk decoder are not the same"
        print("Test 1 passed")
    except AssertionError as e:
        print("Test 1 failed: ", e)
    
    print("Test 2: With interpolation steering on layer L, all output before L should be the same.")
    
    str_array_steering, values_steering, indices_steering = topk_decoder_with_intervention(
        model_name,
        model,
        tokenizer,
        prompt,
        layer_list,
        k,
        steering_method=interpolation_steering,
        steering_strength=steering_strength,
        steering_layer=steering_layer,
        save_dir="data/thinking",
        save_name_suffix="_interpolation_steering_2e-1_layer_20_paris",
        save_data=False
    )
    print_shape(values_steering)
    print_shape(values_topk_decoder)
    try:
        assert t.allclose(values_steering[:2], values_topk_decoder[:2]), "The topk decoder with intervention and the topk decoder are not the same"
        assert t.allclose(indices_steering[:2], indices_topk_decoder[:2]), "The topk decoder with intervention and the topk decoder are not the same"
        print("Test 2 passed")
    except AssertionError as e:
        print("Test 2 failed: ", e)
        
    print("Test 3: With interpolation steering on layer L, all output after L should be the different.")
    try:
        assert not t.allclose(values_steering[steering_layer+1:], values_topk_decoder[steering_layer+1:], atol=1e-3), "The intervention has been ignored and all output is the same"
        print("Test 3 passed")
    except AssertionError as e:
        print("Test 3 failed: ", e)
        
    print(values_topk_decoder-values_steering)
    
    

if __name__ == "__main__":
    # debug_get_probs_all_layers_with_intervention()
    # debug_topk_decoder_with_intervention()
    trail()
import torch as t
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM
from utils import load_model_and_tokenizer, print_model_info, print_shape
import os
import pickle
os.chdir("/mnt/users/clin/workspace/mathematics_transformer")
os.makedirs("data/evolution", exist_ok=True)
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import time
import numpy as np
import einops
from jaxtyping import Float
from dataclasses import dataclass
import json

model_name = "albert"
model, tokenizer = load_model_and_tokenizer(model_name)
# assert isinstance(model, LanguageModel), "model is not a `LanguageModel` from nnsight"





def test_model(model, tokenizer, prompt):
    """
    Test the model's generation capabilities with a given prompt.
    
    This function loads a language model and tokenizer, then generates text continuation
    based on the provided prompt. It demonstrates the model's ability to generate coherent
    text by sampling from the model's output distribution.
    
    Args:
        model: A LanguageModel instance from nnsight for text generation
        tokenizer: The tokenizer corresponding to the model for text encoding/decoding
        prompt (str): The input text prompt to continue from
        
    Returns:
        None: The function prints the generated text to stdout
        
    Example:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2-xl")
        >>> test_model(model, tokenizer, "Once upon a time")
        # Outputs the generated continuation of the prompt
    """
    
    print_model_info(model)
    # Define a simple mathematical prompt


    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(f"Prompt: {prompt}")
    print(f"Input IDs: {input_ids}")
    print(f"Decoded: {tokenizer.decode(input_ids[0])}")

    # Ask the model to continue the story
    with model.generate(input_ids, max_new_tokens=150, do_sample=True, temperature=0.7, top_p=0.9) as gen:
        ft_out = model.generator.output[0].save()
    print(f"Generated: {tokenizer.decode(ft_out)}")

def compute_offdiag_similarity(resid_stream_vector: Float[t.Tensor, "seq_len d_model"]):
    norms = t.norm(resid_stream_vector, dim=1, keepdim=True)
    normalized_resid_stream_vector = resid_stream_vector / norms
    similarity_matrix = normalized_resid_stream_vector @ normalized_resid_stream_vector.T
    return similarity_matrix

@dataclass
class GenerationSettings:
    # For the model generators
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


def get_prompt(model: LanguageModel, tokenizer: AutoTokenizer, instruction: str, generation_settings: GenerationSettings = GenerationSettings(), print_info: bool = True):
    """
    Get an input for the model and tokenizer to generate an in-distribution prompt. 
    """
    with t.no_grad():
        input_ids = tokenizer.encode(instruction, return_tensors="pt").cuda()
        with model.generate(
            input_ids,
            max_new_tokens=generation_settings.max_new_tokens,
            do_sample=generation_settings.do_sample, 
            temperature=generation_settings.temperature, 
            top_p=generation_settings.top_p
        ) as tracer:
            output_ids = model.generator.output[0].save()
    if print_info:
        print(f"Generated text: {tokenizer.decode(output_ids)}")

    return tokenizer.decode(output_ids)

def get_random_tokens(tokenizer: AutoTokenizer, generation_settings: GenerationSettings = GenerationSettings(), print_info: bool = True):
    """
    Baseline of getting a prompt with random text
    """
    num_generated_tokens = generation_settings.max_new_tokens
    random_indices = t.randint(0, tokenizer.vocab_size, (num_generated_tokens,))
    random_tokens = tokenizer.decode(random_indices)
    if print_info:
        print(f"Random tokens: {random_tokens}")
        
    return random_tokens

def gather_activations(
    model: LanguageModel, 
    tokenizer: AutoTokenizer, 
    prompt: str,
    print_info: bool = True, 
    name_str: str = "",
    save_activations: bool = False,
    save_similarity_matrices: bool = True,
):
    if print_info:
        print_model_info(model)

    # save the intermediate activations
    activations = []
    similarity_matrices = []

    with t.no_grad():
        with model.trace(prompt) as tracer:
            for i in range(model.config.n_layer):
                activation = model.transformer.h[i].output[0][0].save()
                activations.append(activation)
                
        activations = t.stack(activations)
        if print_info:
            print_shape(activations)
        
    for i in range(activations.shape[0]):
        similarity_matrix = compute_offdiag_similarity(activations[i])
        similarity_matrices.append(similarity_matrix)

    similarity_matrices = t.stack(similarity_matrices)
    print_shape(similarity_matrices)
    similarity_matrices = similarity_matrices.to(t.float32).to("cpu")
    if save_similarity_matrices:
        t.save(similarity_matrices, f"data/evolution/similarity_matrices{name_str}.pt")
        print(f"Saved {similarity_matrices.shape} similarity matrices to data/evolution/similarity_matrices{name_str}.pt")
    if save_activations:
        t.save(activations, f"data/evolution/activations{name_str}.pt")
        print(f"Saved {activations.shape} activations to data/evolution/activations{name_str}.pt")
    return activations, similarity_matrices

    
def generate_name_str(model_name: str, prompt_label: str):
    return f"_{model_name}_{prompt_label}"

def gather_activations_all_prompts_and_models(
    model_names: list[str],
    prompts_with_labels: list[dict[str, str]],
    print_info: bool = True,
    save_activations: bool = False,
    save_similarity_matrices: bool = False,
    save_data_dict: bool = True,
):
    
    data_dict = {}
    for model_name in model_names:
        model, tokenizer = load_model_and_tokenizer(model_name)
        assert isinstance(model, LanguageModel), "model is not a `LanguageModel` from nnsight"
        if print_info:
            print(f"Gathering activations for {model_name}...")
        
        data_dict[model_name] = {}
        for i, (prompt_label, prompt) in enumerate(prompts_with_labels.items()):
            print(f"Gathering activations for {model_name} and {prompt_label}...")
            name_str = generate_name_str(model_name, prompt_label)
            activations, similarity_matrices = gather_activations(
                model, 
                tokenizer, 
                prompt, 
                name_str=name_str,
                save_activations=save_activations,
                save_similarity_matrices=save_similarity_matrices
            )
            
            data_dict[model_name][prompt_label] = {
                "activations": activations,
                "similarity_matrices": similarity_matrices
            }
    
    if save_data_dict:
        t.save(data_dict, f"data/evolution/data_dict.pth")
        print(f"Saved data_dict to data/evolution/data_dict.pth")
            
    return data_dict

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer("gpt2-xl")
    
    # Obtain prompts and store them in the prompt list
    # in-distribution prompts
    prompt_labels = []
    prompts = []
    instruction = "Once upon a time, there was a "
    prompt = get_prompt(model, tokenizer, instruction)
    # Write the prompt to prompt_list.json
    prompt_labels.append("gpt2xl-indist")
    prompts.append(prompt)
    
    # random baseline prompt
    prompt = get_random_tokens(tokenizer)
    prompt_labels.append("gpt2xl-random")
    prompts.append(prompt)
    
    # Load existing prompts
    try:
        with open("prompt_list.json", "r") as f:
            prompt_list = json.load(f)
    except FileNotFoundError:
        prompt_list = {}
    
    for prompt_label, prompt in zip(prompt_labels, prompts):
        prompt_list[prompt_label] = prompt
    # Save back to file
    with open("prompt_list.json", "w") as f:
        json.dump(prompt_list, f, indent=4)
        
    # Load the updated prompt list
    with open("prompt_list.json", "r") as f:
        prompt_with_labels = json.load(f)
    
    # activations, similarity_matrices = gather_activations(model, tokenizer, prompt)
    data_dict = gather_activations_all_prompts_and_models(
        model_names=["albert"],
        prompts_with_labels=prompt_with_labels,
        save_activations=False,
        save_similarity_matrices=False,
        save_data_dict=True
    )
    




    


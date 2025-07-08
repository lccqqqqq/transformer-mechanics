import torch as t
from nnsight import LanguageModel
from utils import load_model_and_tokenizer
from jaxtyping import Float
from torch import Tensor
from jaxtyping import Int
from typing import Callable
import json
from itertools import product
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

def direct_steering(
    target_vector: Float[Tensor, "... d_model"],
    original_vector: Float[Tensor, "... d_model"],
    steering_strength: float = 0.1,
):
    """
    Note:
        This method can lead to large magnitude changes if the target vector
        has high magnitude or if steering_strength is large. Consider using
        interpolation_steering for more controlled modifications.
    """
    new_vector = original_vector + steering_strength * target_vector
    return new_vector

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
    new_vector = original_vector + steering_strength * (random_vector / t.norm(random_vector) - original_vector)
    return new_vector

def run_steering(
    model: LanguageModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    steering_strength: float = 0.1,
    steering_method: Callable = interpolation_steering,
):
    model.eval()
    decoder = model.lm_head.weight.data
    
    # a clean run
    intervention_output = []
    with t.no_grad():
        with model.trace() as tracer:            
            for layer in tqdm(model.transformer.h):
                # decode the layer residual stream
                with tracer.invoke(prompt) as invoker:
                    layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))
                    layer_output_tokens = t.argmax(layer_output, dim=-1)
                    
                    target_vectors = decoder[layer_output_tokens]
                    layer.output[0][0] = steering_method(target_vectors, layer.output[0][0], steering_strength)
                    
                    # get overall output
                    # make sure the output is a true probability distribution
                    output = t.nn.functional.softmax(model.lm_head.output[0], dim=-1).save()
                    intervention_output.append(output)
            
            with tracer.invoke(prompt) as invoker:
                # make sure the output is a true probability distribution
                original_output = t.nn.functional.softmax(model.lm_head.output[0], dim=-1).save()
    
    intervention_output = [original_output] + intervention_output
    intervention_output = t.stack(intervention_output)
    t.save(intervention_output, f"data/thinking/{steering_method.__name__}_{int(steering_strength * 10)}e-1.pt")
    
    return intervention_output
    

def degradation_metric(prompt: str, intervention_output: Float[Tensor, "n_intervention ctx d_vocab"], tokenizer: AutoTokenizer, k: int = 10, device: str = "cuda"):
    ground_truth_tokens = t.tensor(tokenizer.encode(prompt)[1:]).unsqueeze(0).unsqueeze(-1).to(device)
    # the shape: (n_intervention + 1, ctx, k) where the first dimension is the unsteered output.
    values, indices = t.topk(intervention_output[:, :-1, :], k=k, dim=-1)
    matching_prob_within_topk = t.where(
        indices == ground_truth_tokens,
        values,
        t.zeros_like(values),
    )
    # sum along the top k dimension
    # the shape: (n_intervention + 1, ctx)
    matching_prob_within_topk = matching_prob_within_topk.sum(dim=-1).mean(dim=-1)
    return matching_prob_within_topk

def main():
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.float32)
    device = "cuda"
    # some text in distribution 
    prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    
    n_steering_strength = 10
    k = 10
    
    steering_strengths = {
        interpolation_steering: t.linspace(0, 1, n_steering_strength),
        random_steering: t.linspace(0, 1, n_steering_strength),
    }
    accuracy_measures = {}
    for steering_method, steering_strength_set in tqdm(steering_strengths.items()):
        print(f"Running {steering_method.__name__}")
        output = []
        for steering_strength in steering_strength_set:
            intervention_output = run_steering(
                model,
                tokenizer,
                prompt,
                steering_strength=steering_strength,
                steering_method=steering_method,
            )
            output.append(degradation_metric(prompt, intervention_output, tokenizer, k=k, device=device))

        accuracy_measures[steering_method] = t.stack(output)
    
    # Stack all values in the dict to a list, then stack to a tensor
    all_values = list(accuracy_measures.values())
    stacked_accuracy = t.stack(all_values)
    # save the accuracy measures
    t.save(stacked_accuracy, "data/thinking/accuracy_measures.pt")
    return stacked_accuracy

def main_batch_prompts():
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.float32)
    device = "cuda"
    ds = load_dataset("stas/openwebtext-10k")
    ds = ds.shuffle(seed=42)
    num_examples = 100
    
    for i in tqdm(range(num_examples)):
        text = ds['train'][i]['text']
        input_ids = tokenizer(text, max_length=model.config.n_positions, truncation=True, return_tensors="pt")["input_ids"]
        
        
    
    

def debug():
    model_name = "gpt2-xl"
    model, tokenizer = load_model_and_tokenizer(model_name, device="cuda", torch_dtype=t.float32)
    device = "cuda"
    
    ds = load_dataset("stas/openwebtext-10k")
        

# Todo: Batch process data for different models 


if __name__ == "__main__":
    # model_name = "gpt2-xl"
    # device = "cuda"
    # prompt = "One day, she was walking down the street when she was approached by a man who asked her out on a date."
    # intervention_output = run_steering(model_name, device, prompt, steering_strength=0.8)
    # # intervention_output = t.load("data/thinking/intervention_output.pt", map_location="cpu")
    # from transformers import AutoTokenizer
    # from utils import MODEL_LIST
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_LIST[model_name])
    # print(degradation_metric(prompt, intervention_output, tokenizer, k=10, device=device))
    main()


from thinking import top_k_metric
from utils import load_model_and_tokenizer, print_shape
import torch as t
import json

with open("prompt_list.json", "r") as f:
    prompt_list = json.load(f)

model, tokenizer = load_model_and_tokenizer("gpt2-xl", device="cuda", format="nns")

# The output of the first layer
with t.no_grad():
    with model.trace(prompt_list["gpt4o-normal"]) as tracer:
        layer_0_output = model.transformer.h[0].output[0][0].save()
        decoded_output = model.lm_head(model.transformer.ln_f(layer_0_output)).save()

print_shape(layer_0_output)
print_shape(decoded_output)

# Use weights to calculate explicitly the outcome
# with t.no_grad():
#     layer_0_output_explicit = model.transformer.wte.weight.data[
#         tokenizer.encode(prompt_list["gpt4o-normal"])
#     ] + model.transformer.wpe.weight.data

# print_shape(layer_0_output_explicit)
# assert t.allclose(layer_0_output, layer_0_output_explicit)


# %%
from utils import load_model_and_tokenizer

model_name = 'gpt2-small'
model, tokenizer = load_model_and_tokenizer(model_name, device="cpu", format='nns')

# %%

W_E = model.transformer.wte.weight.data.detach()
W_U = model.lm_head.weight.data.detach()

# %%

word_king = tokenizer.encode(" king")
word_queen = tokenizer.encode(" queen")
word_man = tokenizer.encode(" man")
word_woman = tokenizer.encode(" woman")

print(W_E[word_king].norm())
print(W_E[word_man].norm())
print(W_E[word_queen].norm())
print(W_E[word_woman].norm())

# %%


imperfect_analogy = (W_E[word_king] - W_E[word_man] - W_E[word_queen] + W_E[word_woman]).norm()
print(imperfect_analogy)
imperfect_analogy_direction_only = (W_E[word_king]/W_E[word_king].norm() - W_E[word_man]/W_E[word_man].norm() - W_E[word_queen]/W_E[word_queen].norm() + W_E[word_woman]/W_E[word_woman].norm()).norm()
print(imperfect_analogy_direction_only)

# %% try a decoder approach
alpha = 1.2

imperfect_queen = W_E[word_king] - alpha * W_E[word_man] + alpha * W_E[word_woman]
probs = model.lm_head(imperfect_queen)
print(tokenizer.decode(probs.argmax()))

probs_direction_only = model.lm_head(model.transformer.ln_f(W_E[word_king]/W_E[word_king].norm() - W_E[word_man]/W_E[word_man].norm() - W_E[word_queen]/W_E[word_queen].norm() + W_E[word_woman]/W_E[word_woman].norm()))
print(tokenizer.decode(probs_direction_only.argmax()))

# %%




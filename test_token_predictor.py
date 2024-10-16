"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_token_predicter import GPTConfig, GPT_token_predictor

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT_token_predictor(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT_token_predictor.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

# look for the meta pickle in case it is available in the dataset folder
input_text = input("What would you like to tokenize? ")

char_tokens = [ord(char) for char in input_text]
char_tokens = torch.tensor([char_tokens], dtype=torch.long, device=device)

model_out = model(char_tokens)[0].argmax(dim=-1)[0]

enc = tiktoken.get_encoding("gpt2")
true_tokens = enc.encode_ordinary(input_text)
true_token_locs = enc.decode_with_offsets(true_tokens)[1]

print(len(char_tokens[0]))
print(true_token_locs)

true_out = torch.zeros(len(char_tokens[0]), dtype=torch.int)
true_out[true_token_locs] = 1

print("Model output:")
print(model_out)
print("True output:")
print(true_out)

# Color code the output text
model_out = model_out.tolist()
true_out = true_out.tolist()
input_text = list(input_text)

input_text_true = input_text.copy()
input_text_model = input_text.copy()

for i in range(len(input_text)):
    if true_out[i] == 1:
        input_text_true[i] = '\033[92m|' + input_text[i] + '\033[0m'
    if model_out[i] == 1:
        input_text_model[i] = '\033[92m|' + input_text[i] + '\033[0m'

print("Model output:")
print(''.join(input_text_model))
print("True output:")
print(''.join(input_text_true))


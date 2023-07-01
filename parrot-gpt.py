# Install PyTorch
!conda install pytorch torchvision torchaudio -c pytorch -c nvidia -c apple -y

# Install HuggingFace Transformers
!conda install -c huggingface transformers -y

# Import necessary packages
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# Get the GPT-Neo model
model_name = "EleutherAI/gpt-neo-2.7B"
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Check GPU availability and VRAM
free_vram = 0.0
if torch.cuda.is_available():
    from pynvml import *
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    free_vram = info.free / 1048576000
    print("There is a GPU with " + str(free_vram) + "GB of free VRAM")

# Move the model to GPU if available
if model_name == "EleutherAI/gpt-neo-2.7B" and free_vram > 13.5:
    use_cuda = True
    model.to("cuda:0")
elif model_name == "EleutherAI/gpt-neo-1.3B" and free_vram > 7.5:
    use_cuda = True
    model.to("cuda:0")
else:
    use_cuda = False

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Get user input for prompt and output length
prompt = str(input("Please enter a prompt: "))
output_length = int(input("How long should the generated output be? "))

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Move input_ids to GPU if available
if use_cuda:
    input_ids = input_ids.cuda()

# Generate text with the model
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=output_length)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)

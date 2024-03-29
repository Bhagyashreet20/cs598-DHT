
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import file_utils

print("downloading tokenizer")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
print("done with downloading tokenizer")
print('downloading base t5 model')
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
print("downloaded base t5 model")

cache_dir = file_utils.default_cache_path
print(f"Current Transformers cache directory: {cache_dir}")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))




# GPU version
# import torch

# # Check if CUDA (GPU support) is available
# if torch.cuda.is_available():
#     # Tell PyTorch to use the GPU
#     device = torch.device("cuda")
#     print("Using GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print("Using CPU")

# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Load tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

# # Move model to the appropriate device (GPU or CPU)
# model = model.to(device)

# # Prepare input text
# input_text = "translate English to German: How old are you?"
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# # Move input_ids to the same device as the model
# input_ids = input_ids.to(device)

# # Generate output
# outputs = model.generate(input_ids)

# # Decode and print the generated text
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

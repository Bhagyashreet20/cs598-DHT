
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import file_utils
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE",device)

print("******downloading tokenizer******")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
print("************done with downloading tokenizer*************")
print('***********downloading base t5 model******************')
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

print("******************downloaded base t5 model***********************")
model = model.to(device)
print("moved the model to",model.device)
cache_dir = file_utils.default_cache_path
print(f"Current Transformers cache directory: {cache_dir}")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_ids = input_ids.to(device)
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
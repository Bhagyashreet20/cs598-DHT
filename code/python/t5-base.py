
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import file_utils

print("******downloading tokenizer******")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
print("************done with downloading tokenizer*************")
print('***********downloading base t5 model******************')
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
print("******************downloaded base t5 model***********************")

cache_dir = file_utils.default_cache_path
print(f"Current Transformers cache directory: {cache_dir}")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
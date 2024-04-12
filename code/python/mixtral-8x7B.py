from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch 



def model_size_in_bytes(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.data.nelement() * param.data.element_size()
    return param_size

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_path = "/scratch/bcng/cs598-DHT/models"
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is",device)
# model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
print("loading model")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)
print("quantized_config loaded")
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",cache_dir = model_path, quantization_config=quantization_config)
#model= model.to(device)
print("model is running on",model.device,model)
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
inputs = inputs.to(device)
outputs = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
model_memory = model_size_in_bytes(model)
print(f"Model memory consumption (in bytes): {model_memory}")

#bring up mixtral and run some inference on it
#then create dataset and then augment using the prompts
    #this includes figuringout using mixtral with my prompt and storing the samples based on the dataset developed

    #TASKS 1. figure out how to run the mixtral
    #      2. generate dataset with mission id information in it.




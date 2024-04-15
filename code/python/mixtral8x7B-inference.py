from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import torch 
import json




model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model_path = "/scratch/bcng/cs598-DHT/models"
dataset_path ='/projects/bcng/cs598-DHT/teach_data/edh_instances/processed_data.json'
train_set = '/projects/bcng/cs598-DHT/teach_data/edh_instances/train/'
val_set = '/projects/bcng/cs598-DHT/teach_data/edh_instances/val_seen/'


def gen_prompt(instruction,objects):

    messages = [
        {"role":"system", "content":f'''Act like an embodied NLP agent  in a controlled environmnet where your  task is to generate sequence of actions based on dialog history given.
                                      You will be given with the demonstration examples as a reference to generate the output.
                                      There are two roles in the environment: Driver and Commander. You are Driver and you will always predict the action sequence for Commander’s commands. 
                                      Below are the set of actions that can be taken in the environment.
                                      ['Stop','Move to', 'Forward','Backward','Turn Left','Turn Right','Look Up','Look Down','Pan Left','Pan Right','Move Up','Move Down','Double Forward','DoubleBackward', 'Navigation', 'Pickup','Place','Open', 'Close', 'ToggleOn','ToggleOff','Slice','Dirty','Clean','Fill','Empty','Pour', 'Break','BehindAboveOn','BehindAboveOff','OpenProgressCheck’]
                                      Below are the objects in the environment:
                                      {objects}
                                      Input has the following format:
                                      Please predict the action sequence for the given command. Some examples are:
                                     [[N examples]]
                                      Now, please predict action sequence for the following command.
                                     “For the command [[command]], the predicted action sequence were: “
                                     
                                      And given examples are of the following format:   
                                     “For the command [[command]], the predicted action sequence were: [action sequence]”
                                      Please note the following:
                                      1. Actions like Pickup, Place, Open, Close etc will be associated with objects and remaining actions will not have objects associated with them, In such cases output only the action in the predicted sequence
                                      2. Strictly adhere to the given format and generate ONLY the predicted sequence and do not explain the answer.

                                        '''},
        {"role": "user", "content": f'''Please predict the action sequence for the given command. Some examples are :
                                        "For the command 'please put the two newspapers from the self onto a single table', the predicted action sequence were: [‘Pickup Newspaper', 'Turn Right', 'Place CoffeeTable', 'Turn Left', 'Forward', 'Forward', 'Pan Left', 'Pickup Newspaper', 'Turn Right', 'Place CoffeeTable']. " 

                                        "For the command 'make 2 slices of lettuce', the predicted action sequence were: ['Turn Right', 'Turn Right', 'Turn Right', 'Turn Right', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Open Fridge', 'Pan Left', 'Forward', 'Forward', 'Forward', 'Pickup Lettuce', 'Turn Left', 'Turn Left', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Turn Right', 'Pan Left', 'Pan Left', 'Turn Left', 'Place CounterTop']. "
                                        Now, please predict the following sequence.
                                        "For the command '{instruction}',  the predicted sequence of actions was: "
                                        '''},
       
    ]

    return messages


def gen_object_list(filename):
    with open(f"{train_set}/{filename}",'r') as file:
        data = json.load(file)

    objects = data.get("init_state_diff", {}).get("objects", {})
    object_str = ''
    for object_name, attributes in objects.items():
            object_type, *coordinates = object_name.split('|')
            object_str += object_type + " "
            if 'receptacleObjectIds' in attributes:
                for receptacle_object in attributes['receptacleObjectIds']:
                    receptacle_object_type, *receptacle_coordinates = receptacle_object.split('|')
                    object_str += receptacle_object_type + " "
    return object_str




print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is",device)



quantization_config = BitsAndBytesConfig(load_in_8bit=True,bnb_8bit_compute_dtype=torch.float16)
print("quantized_config loaded")


print("loading model")
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",cache_dir = model_path, quantization_config=quantization_config)





    

with open(dataset_path, 'r') as file:
    data = json.load(file)

count=1
for file_key, content in data.items():
    print(f'processing file:{file_key}')
    for step, details in content.items():
        instruction = details['instruction']
        print(f"running model inference for:{instruction}")
        objects = gen_object_list(file_key)
        messages = gen_prompt(instruction,objects)
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model.generate(inputs, max_new_tokens=250)
        result=tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text for instruction:{instruction} is : {result}")

        if count==1:
            break
        #TODO:now need to write the resutl to a json and store it
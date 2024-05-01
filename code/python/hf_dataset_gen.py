#creating huggingface dataset
# you can replace the keys in different combinations.
import json
train_data={
   'instructions':[],
   'actions':[]
}
with open('./datasets/augmented_dataset.json', 'r') as file:
        data= json.load(file)

for file_key, content in data.items():
    print(f'processing file: {file_key}')
    for step, details in content.items():
       train_data["instructions"].append(data[file_key][step]["paraphrased_instruction"])
       train_data["actions"].append(str(data[file_key][step]["augmented_actions"][0]))

with open('hf_augmented_dataset.json', 'w') as file:
    json.dump(train_data, file, indent=4)
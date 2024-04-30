import json

import re
import os

from groq import Groq
import time

#client = OpenAI()

client = Groq()

'''
Assumes teach-dataset is already in place and have access to train dataset instances. replace the path accordingly
'''
dataset_path ='/projects/bcng/cs598-DHT/teach_data/edh_instances/processed_data.json'
train_set = '/projects/bcng/cs598-DHT/teach_data/edh_instances/train/'


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
                                      3. you must generate action sequence as list of strings delimited by comma like this : [‘Pickup Newspaper', 'Turn Right', 'Place CoffeeTable', 'Turn Left', 'Forward', 'Forward', 'Pan Left', 'Pickup Newspaper', 'Turn Right', 'Place CoffeeTable'] 

                                        '''},


        {"role":"user","content":f'''    Please predict the action sequence for the given command. Some examples are :
                                        "For the command 'please put the two newspapers from the self onto a single table', the predicted action sequence were: [‘Pickup Newspaper', 'Turn Right', 'Place CoffeeTable', 'Turn Left', 'Forward', 'Forward', 'Pan Left', 'Pickup Newspaper', 'Turn Right', 'Place CoffeeTable']" 

                                        "For the command 'make 2 slices of lettuce', the predicted action sequence were: ['Turn Right', 'Turn Right', 'Turn Right', 'Turn Right', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Open Fridge', 'Pan Left', 'Forward', 'Forward', 'Forward', 'Pickup Lettuce', 'Turn Left', 'Turn Left', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Turn Right', 'Pan Left', 'Pan Left', 'Turn Left', 'Place CounterTop']"
                                        Now, please predict the following sequence.
                                        "For the command '{instruction}',  the predicted sequence of actions was: "

'''} ]
   




    return messages


def gen_object_list(filename):
    with open(f"{train_set}/{filename}",'r') as file:
        data = json.load(file)

    objects = data.get("init_state_diff", {}).get("objects", {})
    object_list= set()
    for object_name, attributes in objects.items():
            object_type, *coordinates = object_name.split('|')
            object_list.add(object_type)
            if 'receptacleObjectIds' in attributes:
                for receptacle_object in attributes['receptacleObjectIds']:
                    receptacle_object_type, *receptacle_coordinates = receptacle_object.split('|')
                    object_list.add(object_type)
    return list(object_list)

def convert_to_list(action_string):
    # Trim the single quotes if they exist at the start and end of the string
    if action_string.startswith("'") and action_string.endswith("'"):
        action_string = action_string[1:-1]
    
    # Split the string on comma to form a list
    actions = action_string.split(", ")
    
    # Strip extra white spaces and quotes from each action
    actions = [action.strip('"') for action in actions]
    
    return actions



def format_response(text):
    #match = re.search(r'\[(.*?)\]', text)
    match = re.search(r'(\[.*?\])', text)
    if match:
        text=match.group(1)
        text= text.replace("'",'"')
        return json.loads(text) # This returns the content within the first pair of brackets found
    elif ',' in text:
        text=text.split(',')
        return text
    
    return None 

def get_action_sequence(messages,count=1):
        request_counter = 0  # Initialize a counter to track the number of requests
        action_sequences=[]
        
        for i in range(count):
            try:
                if request_counter >= 3:  # Check if 50 requests have been made
                    time.sleep(30)  # Sleep for 20 seconds
                    request_counter = 0  # Reset the counter
                response = client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages,
                    temperature=0.5,
                    top_p=1,
                    stop=None,
                )
             
                # action_sequence = convert_to_list(response.choices[0].message.content)
                print("response",response.choices[0].message.content,type(response.choices[0].message.content))
                action_sequence= format_response(response.choices[0].message.content)
                print("formatted-action_sequence",action_sequence,"type", type(action_sequence))
             
                # print(prompt, paraphrase)
                action_sequences.append(action_sequence)
                request_counter += 1 

            except Exception as e:
                print(f"Groq API returned an error: {e}")

                return None
        #print(action_sequences)
        return action_sequences

def main():
    with open(dataset_path, 'r') as file:
            data = json.load(file)
    count=1
    checkpoint_interval = 10
    for file_key, content in data.items():
        print(f'processing file:{file_key}',"file count", count)
        for step, details in content.items():
            if "augmented_actions" not in data[file_key][step]:
                    
                instruction = details['instruction']
                print(f"running model inference for:{instruction}")
                objects = gen_object_list(file_key)
                messages = gen_prompt(instruction,objects)
            
                action_sequences = get_action_sequence(messages)
                if action_sequences is None:
                    break
                data[file_key][step]["augmented_actions"] = action_sequences
                print(f"Generated text for instruction:{instruction} is : {action_sequences}")
                print("count",count)
                if count % checkpoint_interval == 0:
                    # Save a checkpoint
                    with open(f"./checkpoints/checkpoint_{count}.json", "w") as f_out:
                        json.dump(data, f_out, indent=4)
                    print("Checkpoint saved.")
        #         if count==10:
        #              break

        

        # if count==10:
        #      break
        
        count+=1
    with open('final_dataset.json', "w") as f_out:
           json.dump(data, f_out, indent=4)
    


if __name__ == "__main__":
    main()
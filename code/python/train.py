#check that our environment has: transformers, datasets, accelerate!
# Torch imports
import torch
import torch.nn as nn

#data imports
import json
import os

# Huggingface imports
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", max_length = 256).cuda()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

#somehow we have lists of data, instructions and actions
#the path to the split you are tring to process, e.g. train, val, etc
directory_path = "../../teach_data/edh_instances/train"
val_path = "../../teach_data/edh_instances/valid_seen"

definitions = {
    "0" : "Stop",
	  "1" : "Move to",
	  "2" : "Forward",
	  "3" : "Backward",
	  "4" : "Turn Left",
	  "5" : "Turn Right",
	  "6" : "Look Up",
	  "7" : "Look Down",
	  "8" : "Pan Left",
	  "9" : "Pan Right",
	  "10" : "Move Up",
	  "11" : "Move Down",
	  "12" : "Double Forward",
	  "13" : "Double Backward",
	  "300" : "Navigation",
	  "200" : "Pickup",
	  "201" : "Place",
	  "202" : "Open",
	  "203" : "Close",
	  "204" : "ToggleOn",
	  "205" : "ToggleOff",
	  "206" : "Slice",
	  "207" : "Dirty",
	  "208" : "Clean",
	  "209" : "Fill",
	  "210" : "Empty",
	  "211" : "Pour",
	  "212" : "Break",
	  "400" : "BehindAboveOn",
	  "401" : "BehindAboveOff",
	  "500" : "OpenProgressCheck",
	  "501" : "SelectOid",
	  "502" : "SearchObject",
	  "100" : "Text",
	  "101" : "Speech",
	  "102" : "Beep",
}

def get_aligned_sequences(temp, filename = ""):
  aligned_data = []
  dialog = []
  actions = []

  try:
    #temp is assumed to be interactions, pass through each entry
    for t in temp:

      #this if condition handles the commander instructions
      if t["action_id"] == 100 and t["agent_id"] == 0:
        #if we ever have multiple instructions in a mission
        #broken up by action sequences, this should allow us to use them all  
        if len(actions):
          if len(dialog):
              #we add the contiguous commander dialogue to the collected sequence of following actions
            aligned_data += [["||".join(dialog), actions.copy()]]
          #then we reset the dialog and actions lists and handle the next dialogue and actions
          actions = []
          dialog = []
        dialog += [t["utterance"]]

      #this if condition handles the follower actions EXCLUDING DIALOGUE!
      #TODO: figure out how to incorporate follwer dialogue as dialogue history  
      if t["agent_id"] == 1 and t["action_id"] != 100:
        a_id = t["action_id"]
        #these conditionals handle actions that take objects and those that do not
        if a_id < 2 or a_id >= 300:
          actions += [definitions[str(a_id)]]
        elif a_id < 300 and a_id >= 200:
          actions += [" ".join([definitions[str(a_id)], t["oid"][:t["oid"].find("|")]])]
    
    #if we don't end on a commander dialogue then we need to add the last round
    if len(actions):
      if len(dialog):
        aligned_data += [["||".join(dialog), actions.copy()]]
    return aligned_data

  except:
    # print("An error occurred, ignoring this example: {}".format(filename))
    # print("here : ", definitions[str(a_id)], t["oid"])
    # print(t)
    return []

final_dataset = []
print("-------starting dataset collection-------")
#go through all the files in the directory path and add the aligned data to the list
for filename in os.listdir(directory_path):
  with open("{}/{}".format(directory_path, filename)) as f:
    data = json.load(f)
  
  final_dataset += get_aligned_sequences(data["interactions"], filename)

train_data = {'instructions' : [d[0] for d in final_dataset], 'actions' : ["[" + ", ".join(d[1]) + "]" for d in final_dataset]}
train_dataset = Dataset.from_dict(train_data)
print(train_dataset)

valid_dataset = []
for filename in os.listdir(val_path):
  with open("{}/{}".format(val_path, filename)) as f:
    data = json.load(f)
  
  valid_dataset += get_aligned_sequences(data["interactions"], filename)

valid_data = {'instructions' : [d[0] for d in valid_dataset], 'actions' : ["[" + ", ".join(d[1]) + "]" for d in valid_dataset]}
val_dataset = Dataset.from_dict(valid_data)
print(val_dataset)

try:
  v_dataset = val_dataset.shuffle(seed=41)
  t_dataset = train_dataset.shuffle(seed=43)
  print("shuffle methodology worked!")
except:
  print("shuffle methodology didn't work, try another route")

def encode(examples):
  input_temp = tokenizer(examples['instructions'], truncation=True, padding='max_length', max_length = 256)
  output_temp = tokenizer(examples['actions'], truncation=True, padding='max_length', max_length = 128)
  return {"decoder_input_ids" : output_temp["input_ids"], "decoder_attention_mask" : output_temp["attention_mask"],
          "input_ids" : input_temp["input_ids"], "attention_mask" : input_temp["attention_mask"],
          "labels" : output_temp["input_ids"]}


train_tokenized = train_dataset.map(encode, batched=True)
val_tokenized = val_dataset.map(encode, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

#seq2seq trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./trained",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    eval_steps = 1,
    evaluation_strategy="steps",
    predict_with_generate = True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics
)

print("-------starting training-------")
trainer.train()

trainer.save_model("./trained/flan-t5-large_ep5")
print("-------model saved-------")
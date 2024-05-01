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
from datasets import Dataset,load_metric
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from statistics import mean



model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large", max_length = 256).cuda()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

#somehow we have lists of data, instructions and actions
#the path to the split you are tring to process, e.g. train, val, etc
directory_path = "../../teach_data/edh_instances/train"
val_path = "./valid_seen"
test_path = './valid_unseen'

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
def encode(examples):
  input_temp = tokenizer(examples['instructions'], truncation=True, padding='max_length', max_length = 256)
  output_temp = tokenizer(examples['actions'], truncation=True, padding='max_length', max_length = 128)
  return {"decoder_input_ids" : output_temp["input_ids"], "decoder_attention_mask" : output_temp["attention_mask"],
          "input_ids" : input_temp["input_ids"], "attention_mask" : input_temp["attention_mask"],
          "labels" : output_temp["input_ids"]} 


def format(txt):
  lst = txt.split(", ")
  act_list = [a.split(" ") for a in lst]
  return act_list



def act_acc(gt_lst, pred_lst):
  total = max(len([x for xs in gt_lst for x in xs]), len([x for xs in pred_lst for x in xs]))
  correct = 0
  for i in range(min(len(gt_lst), len(pred_lst))):
    for j in range(min(len(gt_lst[i]), len(pred_lst[i]))):
      if gt_lst[i][j] == pred_lst[i][j]:
        correct += 1
  return correct / total



bleu_metric = load_metric('bleu')
bert_metric = load_metric('bertscore')
def compute_metrics(outputs):
  d = {}
  gt = outputs.label_ids
  pred = outputs.predictions
  total_pad = sum(sum(((gt == 0).astype(int) + (pred == 0).astype(int)) == 2))
  total_tokens = gt.shape[0] * gt.shape[1]
  total_match = sum(sum(pred == gt))
  d["token_acc"] = (total_match  - total_pad) / (total_tokens - total_pad)

  gt_dec = tokenizer.batch_decode(torch.from_numpy(gt), skip_special_tokens = True)
  pred_dec = tokenizer.batch_decode(torch.from_numpy(pred), skip_special_tokens = True)
  
  total = 0
  for i in range(gt.shape[0]):
    # print("ground truth : ")
    # print(gt_dec[i])
    # print("predicted : ")
    # print(pred_dec[i])
    total += act_acc(format(gt_dec[i]), format(pred_dec[i]))
  d["action object acc"] = total / gt.shape[0]

  d["bert score"] = mean(bert_metric.compute(predictions=pred_dec, references=gt_dec, lang = 'en')['f1'])
  d["bleu score"] = bleu_metric.compute(predictions=[i.split(" ") for i in pred_dec], references=[[a.split(" ")] for a in gt_dec])['bleu']

  return d 



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
print("-------starting training dataset collection-------")
#go through all the files in the directory path and add the aligned data to the list


with open('./datasets/hf_augmented_dataset.json','r') as file:
      dataset = json.load(file)


train_dataset = Dataset.from_dict(dataset)
print("train_dataset",train_dataset)

print("-------starting  valid seen dataset collection-------")
valid_dataset = []
for filename in os.listdir(val_path):
  with open("{}/{}".format(val_path, filename)) as f:
    data = json.load(f)
  
  valid_dataset += get_aligned_sequences(data["interactions"], filename)

valid_data = {'instructions' : [d[0] for d in valid_dataset], 'actions' : ["[" + ", ".join(d[1]) + "]" for d in valid_dataset]}
val_dataset = Dataset.from_dict(valid_data)
print("val_dataset",val_dataset)

print("-------starting valid unseen dataset collection-------")
test_dataset = []
for filename in os.listdir(test_path):
  with open("{}/{}".format(test_path, filename)) as f:
    data = json.load(f)
  
  test_dataset += get_aligned_sequences(data["interactions"], filename)

test_data = {'instructions' : [d[0] for d in test_dataset], 'actions' : ["[" + ", ".join(d[1]) + "]" for d in test_dataset]}
test_dataset = Dataset.from_dict(test_data)
print("test_dataset",test_dataset)








train_tokenized = train_dataset.map(encode, batched=True)
val_tokenized = val_dataset.map(encode, batched=True)
test_tokenized = test_dataset.map(encode, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

#seq2seq trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./augmented/flan-t5-large-augmented",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=400,
    save_total_limit=2,
    #eval_steps = 1,
    evaluation_strategy="epoch",
    predict_with_generate = True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized.shuffle(seed=41),
    eval_dataset=val_tokenized.shuffle(seed=43),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


print("-------starting training-------")
trainer.train()

trainer.save_model("./augmented/flan-t5-large-augmented_ep2")
print("-------model saved-------")


print("-------evaluation metrics-------")
evaluation = trainer.evaluate(val_tokenized)
print("valid seen eval : ", evaluation)

evaluation = trainer.evaluate(test_tokenized)
print("valid unseen eval : ", evaluation)

evaluation = trainer.evaluate(train_tokenized)
print("train eval : ", evaluation)

print("done")



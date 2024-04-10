import json
import os

#the path to the split you are tring to process, e.g. train, val, etc
directory_path = ""

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

def get_aligned_sequences(temp):
  aligned_data = []
  dialog = []
  actions = []

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
      if a_id < 200 or a_id >= 300:
        actions += [definitions[str(a_id)]]
      elif a_id < 300:
        actions += [" ".join([definitions[str(a_id)], t["oid"][:t["oid"].find("|")]])]
  
  #if we don't end on a commander dialogue then we need to add the last round
  if len(actions):
    if len(dialog):
      aligned_data += [["||".join(dialog), actions.copy()]]
  return aligned_data

final_dataset = []
#go through all the files in the directory path and add the aligned data to the list
for filename in os.listdir(directory_path):
  with open("{}/{}".format(directory_path, filename)) as f:
    data = json.load(f)
  
  final_dataset += get_aligned_sequences(data["interactions"])

import os
import json

directory_path = "../../teach_data/edh_instances/train"

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

def get_aligned_sequences(temp, filename):
    aligned_data = {}
    dialog = []
    actions = []
    step_counter = 0

    try:
        for t in temp:
            if t["action_id"] == 100 and t["agent_id"] == 0:
                if actions:
                    if dialog:
                        aligned_data[f"step_{step_counter}"] = {"instruction": "||".join(dialog), "actions": actions.copy()}
                        step_counter += 1
                    actions = []
                    dialog = []
                dialog.append(t["utterance"])
            
            if t["agent_id"] == 1 and t["action_id"] != 100:
                a_id = t["action_id"]
                if a_id < 2 or a_id >= 300:
                    actions.append(definitions[str(a_id)])
                elif 200 <= a_id < 300:
                    if t.get("oid"):  # Checks if 'oid' exists and is not None
                        oid_part = t["oid"][:t["oid"].find("|")]
                        actions.append(" ".join([definitions[str(a_id)], oid_part]))
        
        if actions and dialog:
            aligned_data[f"step_{step_counter}"] = {"instruction": "||".join(dialog), "actions": actions.copy()}
        
        return aligned_data

    except Exception as e:
        print(f"An error occurred in file {filename}: {e}")
        return {}

final_dataset = {}
print("-------starting dataset collection-------")

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            file_data = get_aligned_sequences(data["interactions"], filename)
            if file_data:
                final_dataset[filename] = file_data

# At this point, `final_dataset` contains the data organized by filenames and steps
output_file_path = "../../teach_data/edh_instances/processed_data.json"
with open(output_file_path, 'w') as json_file:
    json.dump(final_dataset, json_file, indent=4)

print("Dataset collection complete and saved to JSON.")

import json
import nltk

# from nltk.util import ngrams
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import datetime
import logging

# Assuming other necessary imports and functions are already in the script, like nltk.FreqDist
# Set up logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"divergences_{timestamp}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)


def calculate_js_divergence(original_actions, augmented_actions):
    # Convert action lists to frequency distributions
    original_action_dist = nltk.FreqDist(original_actions)
    augmented_action_dist = nltk.FreqDist(augmented_actions)

    # Make sure both distributions have the same length
    all_actions = set(original_action_dist.keys()).union(augmented_action_dist.keys())

    # Check if there are no original or augmented actions
    if len(original_actions) == 0 or len(augmented_actions) == 0:
        return float("nan")  # Return NaN because we cannot compute divergence

    original_action_probs = [
        original_action_dist[action] / len(original_actions)
        if action in original_action_dist
        else 0
        for action in all_actions
    ]
    augmented_action_probs = [
        augmented_action_dist[action] / len(augmented_actions)
        if action in augmented_action_dist
        else 0
        for action in all_actions
    ]

    # Calculate Jensen-Shannon divergence
    js_div = jensenshannon(original_action_probs, augmented_action_probs, base=2)
    return js_div


def calculate_kl_divergence(original_actions, augmented_actions, epsilon=1e-10):
    original_action_dist = nltk.FreqDist(original_actions)
    augmented_action_dist = nltk.FreqDist(augmented_actions)

    # Make sure both distributions have the same length
    all_actions = set(original_action_dist.keys()).union(augmented_action_dist.keys())

    # Check if there are no original or augmented actions
    if len(original_actions) == 0 or len(augmented_actions) == 0:
        return float("nan")  # Return NaN because we cannot compute divergence

    original_action_probs = [
        original_action_dist[action] / len(original_actions)
        if action in original_action_dist
        else 0
        for action in all_actions
    ]
    augmented_action_probs = [
        augmented_action_dist[action] / len(augmented_actions)
        if action in augmented_action_dist
        else 0
        for action in all_actions
    ]

    # Add a small constant to avoid log(0)
    original_action_probs = [
        prob if prob > 0 else epsilon for prob in original_action_probs
    ]
    augmented_action_probs = [
        prob if prob > 0 else epsilon for prob in augmented_action_probs
    ]

    # Calculate KL divergence using entropy function
    kl_div = entropy(original_action_probs, augmented_action_probs)
    return kl_div


def flatten(actions):
    """Flatten a list of lists into a single list."""
    return [action for sublist in actions for action in sublist]


# Start of the script processing
logging.info("Starting the diversity metrics calculation process.")

# Load the augmented data from the JSON file
with open("augmented_dataset_actions.json", "r") as file:
    augmented_data = json.load(file)
logging.info("Loaded augmented data successfully.")

# Initialize metrics
kl_divergences = []
js_divergences = []

divergences = {}
# Iterate over the augmented data and calculate metrics
for file_key, file_data in augmented_data.items():
    divergences[file_key] = {}
    for step_key, step_data in file_data.items():
        logging.info(f"Processing {step_key} in {file_key}.")
        original_actions = step_data.get("actions", [])
        augmented_actions = step_data.get("augmented_actions", [])

        # Check if actions are present
        if not original_actions and not augmented_actions:
            logging.warning(f"No actions found for {step_key} in {file_key}, skipping.")
            continue

        # Flatten the actions lists if they are lists of lists
        if original_actions and isinstance(original_actions[0], list):
            original_actions = flatten(original_actions)
        if augmented_actions and isinstance(augmented_actions[0], list):
            augmented_actions = flatten(augmented_actions)

        # Calculate KL divergence
        kl_div = calculate_kl_divergence(original_actions, augmented_actions)
        if kl_div == float("inf"):
            logging.warning(f"Infinite KL divergence for {step_key} in {file_key}")
            # logging.debug(f"Original action probabilities: {original_action_probs}")
            # logging.debug(f"Augmented action probabilities: {augmented_action_probs}")

        kl_divergences.append(kl_div)

        # Calculate Jensen-Shannon divergence
        js_div = calculate_js_divergence(original_actions, augmented_actions)
        if js_div == 1:
            logging.warning(
                f"Jensen-Shannon divergence is 1 for {step_key} in {file_key}"
            )
            # logging.debug(f"Original action probabilities: {original_action_probs}")
            # logging.debug(f"Augmented action probabilities: {augmented_action_probs}")

        js_divergences.append(js_div)

        # Store the divergences for each step
        divergences[file_key][step_key] = {
            "KL_divergence": kl_div,
            "JS_divergence": js_div,
        }
        logging.info(f"Calculated divergences for {step_key} in {file_key}.")

# Calculate average divergence metrics
average_kl_divergence = sum(kl_divergences) / len(kl_divergences)
average_js_divergence = sum(js_divergences) / len(js_divergences)

# Output the results
logging.info(f"Average KL Divergence: {average_kl_divergence}")
logging.info(f"Average Jensen-Shannon Divergence: {average_js_divergence}")

# Save the evaluation results to a file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

divergences_file = f"divergences_{timestamp}.json"
with open(divergences_file, "w") as file:
    json.dump(divergences, file, indent=4)

logging.info(f"Divergences saved to {divergences_file}")
# End of the script processing
logging.info("Diversity metrics calculation process completed.")

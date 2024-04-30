import anthropic
import os
import json
from dotenv import load_dotenv, find_dotenv
from ratelimit import limits, sleep_and_retry, RateLimitException
from backoff import on_exception, expo

_ = load_dotenv(find_dotenv())

client = anthropic.Client(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Rate Limits for Claude 3 Haiku ( Build Tier 2)
REQUESTS_PER_MINUTE = 900  # max(1000)
TOKENS_PER_MINUTE = 100_000
TOKENS_PER_DAY = 2_500_000

system_prompt = """
<system>
    You are an advanced language model specializing in paraphrasing instructions for household tasks. Your role is to generate clear, concise, and coherent paraphrased versions of instructions given by a Commander to a Follower in a dialogue-based task completion scenario.

    Key objectives:

    1. Preserve the original meaning and intent of each instruction while paraphrasing.
    2. Maintain a consistent and natural tone throughout the paraphrased instructions.
    3. Ensure the paraphrased instructions are easy for the Follower to understand and follow.
    4. Consider the previous instructions as context when paraphrasing the current instruction.
    
    Guidelines:

    - Carefully analyze the context and structure of the original instruction before paraphrasing.
    - Break down multi-part instructions separated by "||" and paraphrase each part independently.
    - Use the previous instructions as context to inform the paraphrasing of the current instruction.
    - Use diverse vocabulary and sentence structures to create variation in the paraphrased instructions.
    - Maintain a clear and logical flow between the paraphrased parts of an instruction.
    - Avoid introducing ambiguity or changing the core meaning of the original instruction.
    
    Additional context:
    The instructions are part of a research project on "LLM Augmented Data for Embodied Task-Oriented Dialogue," which aims to enhance the TEACh dataset through paraphrasing. The paraphrased instructions will be used to train and evaluate models for generating diverse and coherent task-oriented dialogues.

    Output format:
    Please provide the paraphrased instruction as a single string, with multi-part instructions separated by "||" as in the original format. Do not include any additional explanations or commentary.
    </system>
"""


@sleep_and_retry
# @on_exception(expo, RateLimitException, max_tries=8)
@limits(calls=REQUESTS_PER_MINUTE, period=60)
def generate_paraphrases(prompt, count=1):
    paraphrases = []
    input_tokens = 0
    output_tokens = 0
    for i in range(count):
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=0.8,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        paraphrase = response.content[0].text
        paraphrases.append(paraphrase)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    return paraphrases, input_tokens, output_tokens


def load_progress():
    try:
        with open("progress.json", "r") as f:
            progress = json.load(f)
    except FileNotFoundError:
        progress = {"file_key": None, "step_key": None}
    return progress


def save_progress(file_key, step_key):
    progress = {"file_key": file_key, "step_key": step_key}
    with open("progress.json", "w") as f:
        json.dump(progress, f)


# Load the processed data from the JSON file
with open("../../teach-dataset/edh_instances/processed_data.json", "r") as file:
    data = json.load(file)

# Load progress from the progress file
progress = load_progress()
last_file_key = progress["file_key"]
last_step_key = progress["step_key"]

# Generate paraphrases for each instruction in the processed data
augmented_data = {}
token_count = 0
output_file_name = "augmented_instruct_data_claude.json"

start_processing = last_file_key is None

for file_key, file_data in data.items():
    if not start_processing:
        if file_key == last_file_key:
            start_processing = True
        else:
            continue

    augmented_data[file_key] = {}

    for step_key, step_data in file_data.items():
        if not start_processing:
            if step_key == last_step_key:
                start_processing = True
            else:
                continue

        instruction = step_data["instruction"]
        prompt = f"Paraphrase the following instruction while preserving its original meaning and intent:\n\n'{instruction}'\n\nParaphrased instruction:"
        paraphrased_instruction, input_tokens, output_tokens = generate_paraphrases(
            prompt, count=1
        )
        token_count += input_tokens + output_tokens
        print(f"Edh File instance: {file_key}")
        print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

        if token_count >= TOKENS_PER_DAY:
            print("Daily token limit reached. Stopping paraphrase generation.")
            save_progress(file_key, step_key)
            # break

        augmented_data[file_key][step_key] = {
            "original_instruction": instruction,
            "paraphrased_instruction": paraphrased_instruction[0],
            "actions": step_data["actions"],
        }
        save_progress(file_key, step_key)

        # Save the augmented data to a file
        with open(output_file_name, "w") as f_out:
            json.dump(augmented_data, f_out, indent=4)

    if token_count >= TOKENS_PER_DAY:
        print(
            "Daily token limit reached. Stopping paraphrase generation in outer loop."
        )
        # break


print(f"Augmented data saved to {output_file_name}")
print(f"Total tokens used: {token_count}")

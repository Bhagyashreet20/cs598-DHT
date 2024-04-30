import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv, find_dotenv
import backoff

_ = load_dotenv(
    find_dotenv()
)  # Before running this, create .env file with OPENAI_API_KEY set to your OpenAI API key

# Assuming you have set your API key in the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

system_prompt = """
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
"""


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def generate_paraphrases(prompt, count=1):
    paraphrases = []
    input_tokens = 0
    output_tokens = 0
    for i in range(count):
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
        )
        paraphrase = response.choices[0].message.content
        # print(prompt, paraphrase)
        paraphrases.append(paraphrase)

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    return paraphrases, input_tokens, output_tokens


# Load the processed data from the JSON file
with open("../../teach-dataset/edh_instances/processed_data.json", "r") as file:
    data = json.load(file)

# Generate paraphrases for each instruction in the processed data
augmented_data = {}
token_count = 0
output_file_name = "augmented_instruct_data_gpt_4.json"

# Augment the dataset with paraphrased instructions
for file_key, file_data in data.items():
    augmented_data[file_key] = {}

    print(f"Edh File instance: {file_key}")
    for step_key, step_data in file_data.items():
        instruction = step_data["instruction"]
        prompt = f"Paraphrase the following instruction while preserving its original meaning and intent:\n\n'{instruction}'\n\nParaphrased instruction:"
        paraphrased_instruction, input_tokens, output_tokens = generate_paraphrases(
            prompt, count=1
        )
        token_count += input_tokens + output_tokens
        print(f"Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")

        # if token_count >= TOKENS_PER_DAY:
        #     print("Daily token limit reached. Stopping paraphrase generation.")
        #     save_progress(file_key, step_key)
        # break

        augmented_data[file_key][step_key] = {
            "original_instruction": instruction,
            "paraphrased_instruction": paraphrased_instruction[0],
            "actions": step_data["actions"],
        }
        # save_progress(file_key, step_key)

        # Save the augmented data to a file
        with open(output_file_name, "w") as f_out:
            json.dump(augmented_data, f_out, indent=4)

    # if token_count >= TOKENS_PER_DAY:
    #     print(
    #         "Daily token limit reached. Stopping paraphrase generation in outer loop."
    #     )
    # break


print(f"Augmented data saved to {output_file_name}")
print(f"Total tokens used: {token_count}")  # Total tokens used: 9538844

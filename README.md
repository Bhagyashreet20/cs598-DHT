# cs598-DHT

## Introduction

This project investigates the application of Large Language Models (LLMs) for the purpose of data augmentation in task-oriented dialogue systems within simulated environments. The primary objective is to enhance the robustness and performance of dialogue systems by generating diverse and rich training datasets.

## Problem Description

Task-oriented dialogue systems are often limited by the size and diversity of their training datasets. The creation of such datasets is resource-intensive, and the resulting scarcity of data can impede the development of models that generalize well to various tasks and environments. Our project addresses this challenge by leveraging LLMs to augment existing datasets, thereby enriching the data available for training more versatile dialogue systems.

## Motivation

The motivation behind this project is to overcome the constraints of data scarcity and lack of diversity in task-oriented dialogue systems. By using LLMs for data augmentation, we aim to simulate a broader range of dialogues and scenarios that could occur in real-world interactions, without the need for extensive data collection.

## Dataset Details

We utilize the TEACh benchmark dataset for task-oriented dialogues in simulated environments. This dataset includes dialogues that capture human interactions and task completions within these environments, providing a foundation for training and evaluating our models.

## Experiments

### Instruction Paraphrasing and Quality Metrics Evaluation

This README provides instructions on how to run the `instruct_augmented_processed_data_openai.py` script for paraphrasing instructions and then using the `augmented_data_quality_metrics.py` script to evaluate the diversity metrics of the paraphrased instructions.

#### Prerequisites

Before running the scripts, ensure you have the following prerequisites installed:

- Python >= 3.7, <=3.8
- `openai` Python package
- `nltk` Python package
- `sentence-transformers` Python package
- `sklearn` Python package
- `torch` Python package
- `transformers` Python package
- `spacy` Python package
- `gensim` Python package
- `dotenv` Python package
- An API key from OpenAI

#### Setting Up Your Environment

1. Clone the repository containing the scripts.
2. Navigate to the cloned directory.
3. Install the required Python packages using pip:

```bash
pip install openai nltk sentence-transformers scikit-learn torch transformers spacy gensim python-dotenv
```

4. Download the necessary NLTK data:

```python
python
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
```

5. Download the necessary spaCy model:

```bash
python -m spacy download en_core_web_lg
```

6. Create a `.env` file in the same folder where you are running the script and set the OpenAI API key as an environment variable.(Check the sample file at `code/python/.env.txt`. If you are planning to use this file,make sure you remove the file extension `.txt`)

#### Running the Instruction Paraphrasing Script

To run the `instruct_augmented_processed_data_openai.py` script:

1. Ensure your OpenAI API key is set in your environment variables.
2. Run the script using Python:

```bash
python instruct_augmented_processed_data_openai.py
```

The script will load the processed data from the specified JSON file(`datasets/processed_data.json`), generate paraphrases for each instruction, and save the augmented data to an output file named `augmented_instruct_data_gpt_4.json`.

## Using the Output for Diversity Metrics Evaluation

After running the paraphrasing script, you can evaluate the diversity metrics of the paraphrased instructions using the `augmented_data_quality_metrics.py` script:

1. Ensure the `augmented_instruct_data_gpt_4.json` file is in the same directory as the `augmented_data_quality_metrics.py` script.
2. Run the script using Python:

```bash
python augmented_data_quality_metrics.py
```

The script will perform the following evaluations on the paraphrased instructions:

- Calculate BLEU scores
- Calculate semantic similarity using Sentence Transformers and SpaCy
- Calculate perplexity using a pre-trained GPT-2 model
- Calculate linguistic diversity measures (TTR, Lexical Diversity, POS Diversity)
- Calculate KL divergence for action modality
- Perform embedding-based clustering

The results will be saved to two JSON files with timestamps: `evaluation_results_<timestamp>.json` and `clustering_results_<timestamp>.json`.

#### Notes

- The paraphrasing script uses tokens from your OpenAI API quota. Monitor your usage to avoid unexpected charges.
- The evaluation script requires the output file from the paraphrasing script. Ensure the file names match or update the file paths in the script accordingly.
- The scripts may take a significant amount of time to run, depending on the size of your dataset and the performance of your machine.

### Action sequence generation Paraphrasing

Action sequence for a given instruction was generated using mixtral-8x7b-32768 model through [Groq API](https://wow.groq.com/). Use `mixtral8x7B-inference.py` file to generate sequence of actions.

#### Prerequisites

install `groq` package using `pip install groq`

#### Setting Up Your Environment

follow these steps to setup the Groq API key in the environment

```bash
vi ~/.bashrc
export GROQ_API_KEY=<GROQ_API_KEY>
source ~/.bashrc
```

#### Running the action sequence generation script

```bash
python mixtral8x7B-inference.py
```

#### training T5-large model with augmented dataset

1. generate trainable dataset(conformign huggingface dataset format) using `hf_dataset_gen.py`

```bash
python hf_dataset_gen.py
```

2. Train the model using below command. This also computes evalaution metrics discussed in the report.

```bash
python train-augmented.py
```

#### Notes on minxtral inference script

- At most one can intitate n requests simultaneously utilizing total 3000 tokens per minute. Refer [this link](https://console.groq.com/docs/rate-limits) for rate limits .
- The script assumes the checkpoints folder already exists in the current directory. Ensure the file names match or update the file paths in the script accordingly.
- The scripts may take a significant amount of time to run, depending on the size of your dataset and the performance of your machine.

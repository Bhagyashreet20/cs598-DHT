# cs598-DHT

## Introduction

This project investigates the application of Large Language Models (LLMs) for the purpose of data augmentation in task-oriented dialogue systems within simulated environments. The primary objective is to enhance the robustness and performance of dialogue systems by generating diverse and rich training datasets.

## Problem Description

Task-oriented dialogue systems are often limited by the size and diversity of their training datasets. The creation of such datasets is resource-intensive, and the resulting scarcity of data can impede the development of models that generalize well to various tasks and environments. Our project addresses this challenge by leveraging LLMs to augment existing datasets, thereby enriching the data available for training more versatile dialogue systems.

## Motivation

The motivation behind this project is to overcome the constraints of data scarcity and lack of diversity in task-oriented dialogue systems. By using LLMs for data augmentation, we aim to simulate a broader range of dialogues and scenarios that could occur in real-world interactions, without the need for extensive data collection.

## Dataset Details

We utilize the TEACh benchmark dataset for task-oriented dialogues in simulated environments. This dataset includes dialogues that capture human interactions and task completions within these environments, providing a foundation for training and evaluating our models.

## Dataset Preparation

To work with the TEACh dataset, follow these steps:

1. Download the dataset using the provided script:

```bash
teach_download
```

This script will download and extract the necessary files into the default directory `/tmp/teach-dataset`.

2. Set up the environment variables to point to the dataset and other important paths:

```bash
export ET_DATA=/path/to/teach-dataset
export TEACH_ROOT_DIR=/path/to/teach/repo
export ET_LOGS=/path/to/store/checkpoints
export VENV_DIR=/path/to/folder/to/store/venv
export TEACH_SRC_DIR=$TEACH_ROOT_DIR/src
export ET_ROOT=$TEACH_SRC_DIR/guides/modeling/ET
export INFERENCE_OUTPUT_PATH=/path/to/store/inference/execution/files
```

3. Create a virtual environment and install the required dependencies:

```bash
python3 -m venv $VENV_DIR/teach_env
source $VENV_DIR/teach_env/bin/activate
cd $TEACH_ROOT_DIR
pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=$TEACH_SRC_DIR:$ET_ROOT:$PYTHONPATH
```

4. Download the E.T. pretrained checkpoints:

```bash
wget http://pascal.inrialpes.fr/data2/apashevi/et_checkpoints.zip
unzip et_checkpoints.zip
mv pretrained $ET_LOGS/
rm et_checkpoints.zip
```

If the above link doesn't work, you can try this Google drive link to download the checkpoints directly: [google drive](https://drive.google.com/file/d/1RyXDsVKdx4P0i6OQQH2vYTkq-1tQI71z/view?usp=sharing)

5. Preprocess the data to extract image features and process EDH jsons:

```bash
python -m alfred.data.create_lmdb \
with args.visual_checkpoint=$ET_LOGS/pretrained/fasterrcnn_model.pth \
args.data_input=edh_instances \
args.task_type=edh \
args.data_output=lmdb_edh \
args.vocab_path=None
```

For doing this we are using the Slurm script : `code/slurm-scripts/create_lmdb.slurm` and running it on NCSA Delta Cluster. On the cluster, you can run the Slurm script using this command:

```bash
sbatch slurm-scripts/create_lmdb.slurm
```

### Training and Evaluating a Flan-T5 Model
To train a smaller LLM model you can run the `train.py` file from the `slurm-scripts` folder with:

```sbatch train-models.slurm``` 

which should save the trained model in your `scratch` folder. It will automatically run for 5 epochs on the original data, but any training parameters and datasets can be edited within the file.
To evaluate the trained model, yo ucan run the `eval.py` file from the `slurm-scripts` folder with:

```sbatch eval.slurm```

which will print out the desired metrics in the corresponding outputs file.


### Training the E.T. Model

To train the E.T. model on the TEACh dataset, we use the `train_et_model.slurm` SLURM script. This script sets up the necessary environment, loads the required modules, and executes the training command. It also specifies the computational resources needed for the job, such as memory, GPUs, and runtime.

The training process is logged, and the output can be found in the specified log directory. The script will train the model for a specified number of epochs and save the checkpoints to the designated logs directory.

To start the training, submit the SLURM script to your cluster's scheduler:

```bash
sbatch slurm-scripts/train_et_model.slurm
```

### Evaluating the Model

After training, the model's performance can be evaluated using the `run_inference.slurm` SLURM script. This script will run the inference command that loads the trained model and evaluates it on the validation set. It will output the inference results and metrics to the specified output path.

To run the evaluation, submit the SLURM script to your cluster's scheduler:

```bash
sbatch slurm-scripts/run_inference.slurm
```

The inference results will include various performance metrics that are saved to a JSON file. These metrics provide insights into the model's ability to generate sequences of actions that are contextually relevant and feasible within the simulated environment.

#### Additional Notes

- The SLURM scripts are configured for a specific cluster setup. You may need to modify the resource specifications and module loading commands to match your cluster's configuration.
- Ensure that the paths specified in the environment variables and SLURM scripts match the actual locations of your dataset, checkpoints, and output directories.
- Monitor the progress of your SLURM jobs using the `squeue` command and check the output and error logs for any issues that may arise during training or inference.

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

Note: We had also tried paraphrasing the instructions using Claude-3 Haiku model via Anthropic API but we were getting rate limited because the total tokens of our instruction dataset was almost 10 Million which was more than what we could process with our Billing Tier.

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

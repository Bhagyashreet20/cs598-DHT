import matplotlib.pyplot as plt
import numpy as np
import json


# Load the augmented data quality metrics from the JSON file
with open("evaluation_results_20240421_010748.json", "r") as file:
    quality_metrics = json.load(file)

# Calculate average quality metrics across the dataset
average_bleu = np.mean([item["bleu_score"] for item in quality_metrics])
average_semantic_similarity = np.mean(
    [item["semantic_similarity_sentence_transformers"] for item in quality_metrics]
)
average_perplexity = np.mean(
    [item["original_perplexity"] for item in quality_metrics]
)  # Assuming we want the perplexity of the original instructions
average_ttr = np.mean([item["original_ttr"] for item in quality_metrics])
average_lexical_diversity = np.mean(
    [item["original_lexical_diversity"] for item in quality_metrics]
)
average_pos_diversity = np.mean(
    [item["original_pos_diversity"] for item in quality_metrics]
)

# Cherry-picked examples for qualitative comparison
examples = [
    {
        "original_instruction": quality_metrics[0]["original_instruction"],
        "paraphrased_instruction": quality_metrics[0]["paraphrased_instruction"],
        "bleu_score": quality_metrics[0]["bleu_score"],
    },
    # Add more examples as needed
]

import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import FreqDist
from sklearn.cluster import KMeans
from scipy.stats import entropy
import datetime
import spacy
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity

nltk.download("punkt")  # Download the Punkt tokenizer
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


"""
BLEU Score:
BLEU (Bilingual Evaluation Understudy) is a metric commonly used to evaluate the quality of machine-generated text by comparing it to reference text.
You can calculate the BLEU score between the original instructions and the paraphrased instructions to measure their similarity.
A higher BLEU score indicates greater similarity between the original and paraphrased instructions.
"""


def calculate_bleu_score(original_instruction, paraphrased_instruction):
    """
    The function `calculate_bleu_score` calculates the BLEU score between an original instruction and a
    paraphrased instruction in Python.

    :param original_instruction: The `calculate_bleu_score` function you provided calculates the BLEU
    score between an original instruction and a paraphrased instruction. The BLEU score is a metric used
    to evaluate the quality of machine-translated text or text paraphrases
    :param paraphrased_instruction: The `calculate_bleu_score` function calculates the BLEU score
    between an original instruction and a paraphrased instruction. The BLEU score is a metric used to
    evaluate the quality of machine-translated text or paraphrased text by comparing it to one or more
    reference translations
    :return: The function `calculate_bleu_score` returns the BLEU score calculated between the original
    instruction and the paraphrased instruction using the specified parameters.
    """
    reference = [original_instruction.split()]
    candidate = paraphrased_instruction.split()
    smoothing_function = SmoothingFunction().method4
    bleu_score = sentence_bleu(
        reference, candidate, weights=(0.5, 0.5), smoothing_function=smoothing_function
    )
    return bleu_score


"""
Semantic Similarity Measures:
Semantic similarity measures calculate the similarity between two texts based on their meaning rather than exact word overlap.
Some popular semantic similarity measures include:
Cosine Similarity: Calculates the cosine of the angle between two text vectors in a high-dimensional space.
Word Mover's Distance (WMD): Measures the minimum distance required to move the words from one text to another.
Sentence Embeddings: Converts sentences into dense vector representations and calculates the similarity between the vectors using metrics like cosine similarity.
You can use libraries like gensim, spacy, or sentence-transformers in Python to calculate semantic similarity measures."""


def calculate_semantic_similarity(original_instruction, paraphrased_instruction):
    model = SentenceTransformer("paraphrase-distilroberta-base-v1")
    original_embedding = model.encode([original_instruction])
    paraphrased_embedding = model.encode([paraphrased_instruction])
    similarity_score = cosine_similarity(original_embedding, paraphrased_embedding)[0][
        0
    ]
    return similarity_score


def calculate_semantic_similarity_spacy(original_instruction, paraphrased_instruction):
    """
    The function `calculate_semantic_similarity_spacy` uses spaCy to calculate the semantic similarity
    score between an original instruction and a paraphrased instruction.

    :param original_instruction: The `original_instruction` parameter is the original text or
    instruction that you want to compare for semantic similarity with another text. It could be a
    sentence, a paragraph, or any piece of text that you want to analyze for similarity with another
    text
    :param paraphrased_instruction: The `calculate_semantic_similarity_spacy` function you provided
    calculates the semantic similarity between an original instruction and a paraphrased instruction
    using spaCy's pre-trained model `en_core_web_lg`. The function loads the model, processes the
    original and paraphrased instructions, and then calculates the similarity
    :return: The function `calculate_semantic_similarity_spacy` returns a float value representing the
    semantic similarity score between the original instruction and the paraphrased instruction using the
    spaCy library's pre-trained model "en_core_web_lg".
    """
    nlp = spacy.load("en_core_web_lg")
    original_doc = nlp(original_instruction)
    paraphrased_doc = nlp(paraphrased_instruction)
    similarity_score = original_doc.similarity(paraphrased_doc)
    return float(similarity_score)


# def calculate_semantic_similarity_wmd(original_instruction, paraphrased_instruction):
#     """
#     The function `calculate_semantic_similarity_wmd` calculates the Word Mover's Distance (WMD) semantic
#     similarity score between an original instruction and a paraphrased instruction using pre-trained
#     word embeddings.

#     :param original_instruction: The `calculate_semantic_similarity_wmd` function you provided
#     calculates the Word Mover's Distance (WMD) semantic similarity between an original instruction and a
#     paraphrased instruction using spaCy and Gensim's Word2Vec model
#     :param paraphrased_instruction: The `calculate_semantic_similarity_wmd` function you provided
#     calculates the semantic similarity between an original instruction and a paraphrased instruction
#     using Word Mover's Distance (WMD) algorithm
#     :return: The function `calculate_semantic_similarity_wmd` returns a floating-point value
#     representing the semantic similarity score between the original instruction and the paraphrased
#     instruction calculated using the Word Mover's Distance (WMD) algorithm.
#     """
#     nlp = spacy.load("en_core_web_lg")
#     original_doc = nlp(original_instruction)
#     paraphrased_doc = nlp(paraphrased_instruction)

#     original_tokens = [token.text for token in original_doc]
#     paraphrased_tokens = [token.text for token in paraphrased_doc]

#     model = KeyedVectors.load_word2vec_format("path/to/word2vec/model", binary=True)
#     wmd_similarity = WmdSimilarity(paraphrased_tokens, model)
#     similarity_score = wmd_similarity[original_tokens][0]
#     return float(similarity_score)


"""Perplexity:
Perplexity is a measure of how well a language model predicts a given text.
You can use a pre-trained language model (e.g., GPT-2, BERT) to calculate the perplexity of the original instructions and the paraphrased instructions.
Lower perplexity indicates that the language model is better at predicting the text, suggesting higher quality and naturalness.
You can use libraries like transformers in Python to calculate perplexity using pre-trained language models."""


def calculate_perplexity(instruction):
    """
    The function `calculate_perplexity` calculates the perplexity of a given instruction using a
    pre-trained GPT-2 language model.

    :param instruction: It looks like you are trying to calculate the perplexity of a given instruction
    using a pre-trained GPT-2 language model. The `instruction` parameter should be a string containing
    the text for which you want to calculate the perplexity. You can pass the instruction as a string to
    the `calculate
    :return: The function `calculate_perplexity` returns the perplexity value calculated based on the
    input instruction using a pre-trained GPT-2 language model.
    """
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs[0]
        perplexity = torch.exp(loss).item()

    return perplexity


"""Linguistic Diversity Measures:
To assess the linguistic diversity of your dataset, you can calculate various diversity measures:
Type-Token Ratio (TTR): Measures the ratio of unique words (types) to the total number of words (tokens) in a text.
Lexical Diversity: Calculates the ratio of unique lemmas (base forms of words) to the total number of tokens.
Part-of-Speech (POS) Diversity: Measures the distribution and variety of POS tags in the text.
Higher diversity scores indicate a more diverse dataset in terms of vocabulary and linguistic patterns.
You can use libraries like nltk or spacy in Python to perform linguistic analysis and calculate diversity measures."""


def calculate_ttr(instruction):
    tokens = nltk.word_tokenize(instruction)
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens)
    return ttr


def calculate_lexical_diversity(instruction):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(instruction)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    unique_lemmas = set(lemmas)
    lexical_diversity = len(unique_lemmas) / len(tokens)
    return lexical_diversity


def calculate_pos_diversity(instruction):
    tokens = nltk.word_tokenize(instruction)
    pos_tags = pos_tag(tokens)
    pos_freq_dist = FreqDist(tag for _, tag in pos_tags)
    pos_diversity = len(pos_freq_dist) / len(tokens)
    return pos_diversity


def calculate_kl_divergence(original_actions, paraphrased_actions):
    original_action_dist = nltk.FreqDist(original_actions)
    paraphrased_action_dist = nltk.FreqDist(paraphrased_actions)

    original_action_probs = [
        count / len(original_actions) for count in original_action_dist.values()
    ]
    paraphrased_action_probs = [
        count / len(paraphrased_actions) for count in paraphrased_action_dist.values()
    ]

    kl_div = entropy(original_action_probs, paraphrased_action_probs)
    return kl_div


"""Embedding-based Clustering:
Convert the original and paraphrased instructions into vector representations using techniques like word embeddings (e.g., Word2Vec, GloVe) or sentence embeddings (e.g., BERT, RoBERTa).
Apply clustering algorithms (e.g., K-means, DBSCAN) to group similar instructions together based on their vector representations.
Analyze the clusters to assess the diversity of the paraphrased instructions and identify any patterns or similarities.
You can use libraries like scikit-learn or tensorflow in Python for embedding-based clustering."""


def perform_clustering(instructions):
    model = SentenceTransformer("paraphrase-distilroberta-base-v1")
    embeddings = model.encode(instructions)

    num_clusters = 5  # Specify the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)

    cluster_labels = kmeans.labels_
    return cluster_labels


# Load the augmented data from the JSON file
with open("augmented_instruct_data_gpt_4.json", "r") as file:
    augmented_data = json.load(file)

# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Save the evaluation results to a file with timestamp
evaluation_results_file = f"evaluation_results_{timestamp}.json"

results = []

# Iterate over the augmented data and calculate metrics
for file_key, file_data in augmented_data.items():
    for step_key, step_data in file_data.items():
        original_instruction = step_data["original_instruction"]
        paraphrased_instruction = step_data["paraphrased_instruction"]
        original_actions = step_data["actions"]

        print(f"Processing file: {file_key}, step: {step_key}")
        print(f"Original instruction: {original_instruction}")
        print(f"Paraphrased instruction: {paraphrased_instruction}")

        # Calculate BLEU score
        bleu_score = calculate_bleu_score(original_instruction, paraphrased_instruction)
        print(f"BLEU score: {bleu_score}")

        # Calculate semantic similarity using Sentence Transformers
        similarity_score_sentence_transformers = calculate_semantic_similarity(
            original_instruction, paraphrased_instruction
        )
        print(f"Semantic similarity: {similarity_score_sentence_transformers}")

        # Calculate semantic similarity using SpaCy
        similarity_score_spacy = calculate_semantic_similarity_spacy(
            original_instruction, paraphrased_instruction
        )
        print(f"Semantic similarity (SpaCy): {similarity_score_spacy}")

        # # Calculate semantic similarity using Word Mover's Distance
        # similarity_score_wmd = calculate_semantic_similarity_wmd(
        #     original_instruction, paraphrased_instruction
        # )
        # print(f"Semantic similarity (Word Mover's Distance): {similarity_score_wmd}")

        # Calculate perplexity
        original_perplexity = calculate_perplexity(original_instruction)
        paraphrased_perplexity = calculate_perplexity(paraphrased_instruction)
        print(f"Original perplexity: {original_perplexity}")
        print(f"Paraphrased perplexity: {paraphrased_perplexity}")

        # Calculate Type-Token Ratio (TTR)
        original_ttr = calculate_ttr(original_instruction)
        paraphrased_ttr = calculate_ttr(paraphrased_instruction)
        print(f"Original TTR: {original_ttr}")
        print(f"Paraphrased TTR: {paraphrased_ttr}")

        # Calculate Lexical Diversity
        original_lexical_diversity = calculate_lexical_diversity(original_instruction)
        paraphrased_lexical_diversity = calculate_lexical_diversity(
            paraphrased_instruction
        )
        print(f"Original Lexical Diversity: {original_lexical_diversity}")
        print(f"Paraphrased Lexical Diversity: {paraphrased_lexical_diversity}")

        # Calculate POS Diversity
        original_pos_diversity = calculate_pos_diversity(original_instruction)
        paraphrased_pos_diversity = calculate_pos_diversity(paraphrased_instruction)
        print(f"Original POS Diversity: {original_pos_diversity}")
        print(f"Paraphrased POS Diversity: {paraphrased_pos_diversity}")

        # Calculate KL divergence for action modality
        kl_divergence = calculate_kl_divergence(
            original_actions, original_actions
        )  # Placeholder for paraphrased actions
        print(f"KL divergence: {kl_divergence}")
        print()

        result = {
            "file_key": file_key,
            "step_key": step_key,
            "original_instruction": original_instruction,
            "paraphrased_instruction": paraphrased_instruction,
            "bleu_score": float(bleu_score),
            "semantic_similarity_sentence_transformers": float(
                similarity_score_sentence_transformers
            ),
            "semantic_similarity_spacy": float(similarity_score_spacy),
            # "semantic_similarity_wmd": float(similarity_score_wmd),
            "original_perplexity": float(original_perplexity),
            "paraphrased_perplexity": float(paraphrased_perplexity),
            "original_ttr": float(original_ttr),
            "paraphrased_ttr": float(paraphrased_ttr),
            "original_lexical_diversity": float(original_lexical_diversity),
            "paraphrased_lexical_diversity": float(paraphrased_lexical_diversity),
            "original_pos_diversity": float(original_pos_diversity),
            "paraphrased_pos_diversity": float(paraphrased_pos_diversity),
            "kl_divergence": float(kl_divergence),
        }

        results.append(result)


# Write the entire results list to the file as a single JSON object
with open(evaluation_results_file, "w") as f_out:
    json.dump(results, f_out, indent=4)

print(f"Evaluation results saved to {evaluation_results_file}")

# Perform clustering on the entire dataset
all_instructions = [
    step_data["original_instruction"]
    for file_data in augmented_data.values()
    for step_data in file_data.values()
]
all_instructions.extend(
    [
        step_data["paraphrased_instruction"]
        for file_data in augmented_data.values()
        for step_data in file_data.values()
    ]
)

cluster_labels = perform_clustering(all_instructions)

# Save the clustering results to a file with timestamp
clustering_results_file = f"clustering_results_{timestamp}.json"
with open(clustering_results_file, "w") as f_out:
    json.dump(list(zip(all_instructions, cluster_labels)), f_out, indent=4)
print(f"Clustering results saved to {clustering_results_file}")

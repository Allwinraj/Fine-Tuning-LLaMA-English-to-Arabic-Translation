import json
import random
import torch
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score
from transformers import pipeline
from llamafactory.extras.misc import torch_gc

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA-based models

# Model Configuration
model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Prompt Template
PROMPT_TEMPLATE = """{instruction}

Input: {input}

Output: """

# Function to format prompt
def format_prompt(user_input):
    return PROMPT_TEMPLATE.format(instruction="Translate to Arabic:", input=user_input)

# Load test data
def load_test_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

# Perform translation and calculate BLEU & BERTScore
def evaluate_model(test_data):
    predictions = []
    references = []

    for entry in test_data[10:20]:  # Process only first 10 samples
        english_sentence = entry["input"]
        reference_translation = entry["output"]  # Ground truth translation

        formatted_query = english_sentence
        messages = [
            {"role": "system", "content": "You are a translator that converts English text to Arabic. strictly do not add anythin. return only arabic text"},
            {"role": "user", "content": formatted_query},
        ]

        print(f"\n🔍 Translating: {english_sentence}")
        print("Assistant: ", end="", flush=True)
        
        response = pipe(messages, max_new_tokens=256)[0]["generated_text"]
        print(response)

        response = response[-1]["content"]

        # Collect results for BLEU calculation
        predictions.append(response)  # Model-generated translation as tokens
        references.append([reference_translation])  # Reference as tokens

        

    # Calculate BLEU score
    bleu_score = corpus_bleu(references, predictions, smoothing_function=SmoothingFunction().method1)
    # Calculate BERTScore
    P, R, F1 = score([response], [reference_translation], lang="ar")
    print(f"\n🚀 BLEU Score: {bleu_score:.4f}")
    print(f"✅ BERTScore (F1 Similarity): {F1.mean().item():.4f}")

# Load test data and evaluate
test_data = load_test_data("train_english_to_arabic.json")  # Replace with your JSON file
evaluate_model(test_data)

torch_gc()
import json
import random
import torch
import numpy as np
import faiss
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
import os

# Set random seed for reproducibility
SEED = 23
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For CUDA-based models

# Model Configuration
args = dict(
    model_name_or_path="/home/Allwin/Fine_tune/model_finetune_arabic",
    template="llama3",
    finetuning_type="lora",
    quantization_bit=4,
)
chat_model = ChatModel(args)

# Prompt Template
PROMPT_TEMPLATE = """{instruction}

Input: {input}

Output: """

# Function to format prompt
def format_prompt(user_input):
    return PROMPT_TEMPLATE.format(instruction="Translate to Arabic:", input=user_input)

# Load text data from a file and chunk into 100-word segments
def load_and_chunk_text(filename, chunk_size=100):
    with open(filename, "r", encoding="utf-8") as f:
        full_text = f.read().replace("\n", " ")
    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Convert sentences to embeddings
def generate_embeddings(sentences, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings, model

# Create and save an HNSW FAISS index
def create_hnsw_index(embeddings, index_file="rag_index_hnsw.faiss"):
    dimension = embeddings.shape[1]
    hnsw_index = faiss.IndexHNSWFlat(dimension, 32)
    hnsw_index.hnsw.efConstruction = 64
    hnsw_index.hnsw.efSearch = 32
    hnsw_index.add(embeddings)
    faiss.write_index(hnsw_index, index_file)
    return hnsw_index

# Load FAISS index from file
def load_faiss_index(index_file):
    if os.path.exists(index_file):
        hnsw_index = faiss.read_index(index_file)
        print("✅ FAISS index loaded successfully!")
    else:
        print("❌ FAISS index file not found! Please build the index first.")
        hnsw_index = None
    return hnsw_index

# Query the FAISS index
def search_index(query, hnsw_index, text_chunks, model, top_k=1):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = hnsw_index.search(query_embedding, top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results

# Chat loop
def chat_loop(retrieved_texts):
    messages = []
    print("Welcome to the CLI application, use `clear` to remove history, use `exit` to exit.")
    
    for text in retrieved_texts:
        formatted_query = format_prompt(text)
        messages.append({"role": "user", "content": formatted_query})
        print("Assistant: ", end="", flush=True)
        response = ""
        
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        
        print()
    
    torch_gc()

# Main function to build and query the RAG pipeline
def main():
    input_file = "english_sentences.txt"
    index_file = "rag_index_hnsw.faiss"
    text_chunks = load_and_chunk_text(input_file)
    print(f"📌 Loaded {len(text_chunks)} chunks (100 words each).")
    embeddings, model = generate_embeddings(text_chunks)
    if os.path.exists(index_file):
        print("🔄 Loading existing FAISS index...")
        hnsw_index = load_faiss_index(index_file)
    else:
        print("🚀 Creating new FAISS index...")
        hnsw_index = create_hnsw_index(embeddings, index_file)
   
    print("✅ RAG pipeline setup complete! You can now query the system.")
    query = input("\n🔍 Enter your search query: ")
    retrieved_texts = search_index(query, hnsw_index, text_chunks, model)
    print("\n🔹 Top Retrieved Chunks:")
    for i, text in enumerate(retrieved_texts, 1):
        print(f"{i}. {text}")
    chat_loop(retrieved_texts)

if __name__ == "__main__":
    main()

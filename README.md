# Fine-Tuning LLaMA for English-to-Arabic Translation

## Step 1: Prepare Dataset for LLaMA Fine-Tuning

Fine-tuning a language model requires a well-structured dataset. For this purpose, we used an **English-to-Arabic parallel text dataset** from Hugging Face, which contains high-quality sentence pairs useful for machine translation tasks.

- Dataset: [ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts](https://huggingface.co/datasets/ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts)

### Dataset Preparation

To prepare the dataset for fine-tuning, run the following script:

```bash
python llama_dataset_preparation.py
```

#### Why is this step important?
- **Data Splitting**: The dataset is split into training and validation sets to ensure a balanced fine-tuning process.
- **Data Formatting**: The script converts the raw dataset into a structure suitable for LLaMA fine-tuning, ensuring compatibility with the model.
- **Storage in JSON**: The processed dataset is saved in `data/*.json`, making it easy to load during training.

## Step 2: Fine-Tuning the LLaMA Model

Fine-tuning allows us to customize a pre-trained model for our specific use case. Here, we use the **LLaMA 3.2 3B Instruction Model**, which is already optimized for instruction-based tasks and can be further enhanced with domain-specific data.

- Model: [LLaMA-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

Fine-tuning is conducted using the **LLaMA Factory** framework:

- Repository: [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

### Fine-Tuning Configuration

We carefully configure hyperparameters such as learning rate, batch size, and optimizer settings, stored in the `finetuning/` folder.

- The training process is monitored using two key metrics:
  - **Training Loss:** `/finetuning/training_loss.png` (indicates how well the model is learning from the data)
  - **Evaluation Loss:** `/finetuning/training_eval_loss.png` (measures how well the model generalizes to unseen validation data)

#### Why is this step important?
- **Enhancing Model Performance**: Fine-tuning adapts the pre-trained LLaMA model to our English-to-Arabic dataset, improving its translation accuracy.
- **Reducing Overfitting**: Proper evaluation ensures that the model does not memorize the dataset but instead learns to generalize effectively.

## Step 3: Evaluating the Fine-Tuned Model

To verify the effectiveness of our fine-tuned model, we compare it against the base model using **BLEU** and **BERTScore** metrics.

### Evaluation Process

Run the following scripts to evaluate performance:

```bash
python evaluation_finetune_model.py  # Evaluate the fine-tuned model
python evaluation_base_model.py      # Evaluate the base model (LLaMA 3.2 3B)
```

### Evaluation Results

| Model               | BLEU Score | BERTScore (F1 Similarity) |
|---------------------|------------|---------------------------|
| **Base Model**     | 0.3683      | 0.7365                    |
| **Fine-Tuned Model** | 0.5771      | 0.8620                    |

- Some inference results are added
   - **Training Loss:** `/finetuning/result.png`

#### Why is this step important?
- **BLEU Score**: Measures how closely the generated translations match reference translations. A higher BLEU score indicates better accuracy.
- **BERTScore**: Uses deep learning embeddings to assess the semantic similarity between generated and reference translations.
- **Performance Validation**: The results confirm that the fine-tuned model significantly outperforms the base model.

## Step 4: Integrating the Model into a RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** pipeline enhances translation quality by retrieving **contextually relevant** segments before translation.

### RAG Dataset Preparation

Before implementing RAG, we prepare a dataset using:

```bash
python rag_dataset_preparation.py
```

This script processes input text into paragraph-style data, ensuring efficient retrieval.

### RAG Pipeline Design
The RAG pipeline follows these steps:

1. **Read the text file** containing English text.
2. **Chunk the text** into smaller segments, making retrieval more efficient.
3. **Generate embeddings** for each chunk using a pre-trained model.
4. **Store embeddings** in a **FAISS index** for fast lookup.
5. **Retrieve relevant chunks** based on user queries, ensuring contextual accuracy.
6. **Use the fine-tuned model** to translate retrieved English chunks into Arabic.

#### Why is this step important?
- **Contextual Awareness**: Instead of translating isolated sentences, the model has access to semantically relevant information.
- **Improved Accuracy**: The retrieval mechanism reduces translation errors by providing supporting context.
- **Scalability**: Storing embeddings in FAISS allows for efficient real-time retrieval, making this approach suitable for large datasets.

## References

1. **Hugging Face Datasets**
   - [ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts](https://huggingface.co/datasets/ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts)
   - [Meta LLaMA 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. **Fine-Tuning Framework**
   - [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
3. **Evaluation Metrics**
   - [BLEU Score - NLTK](https://www.nltk.org/_modules/nltk/translate/bleu_score.html)
   - [BERTScore](https://github.com/Tiiiger/bert_score)

---

This repository provides all necessary code and configurations for fine-tuning LLaMA on **English-to-Arabic translation** and integrating it into a **RAG pipeline**. 🚀

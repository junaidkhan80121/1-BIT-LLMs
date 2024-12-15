# 1-BIT LLM Project

This project involves implementing a lightweight **1-BIT LLM (Large Language Model)** using a custom configuration of Llama architecture, employing quantization techniques to optimize performance and reduce memory usage.

---

## Project Overview
- **Model Configuration**: Nous-Hermes-Llama-2-7b (customized to a smaller scale).
- **Quantization**: Includes activation and weight quantization for improved efficiency.
- **Backend Framework**: Hugging Face Transformers.
- **Training Dataset**: Cosmopedia-100k-pretrain.

### Key Features
- Implementation of **1-bit quantization** for activations and weights.
- Modified **Llama RMSNorm** and decoder layers.
- Training and monitoring with **WandB**.

---

## Prerequisites
To run this project, ensure you have the following installed:

### Python Libraries
- `transformers`
- `datasets`
- `wandb`
- `accelerate`
- `torch`

Install them via:
```bash
pip install datasets wandb accelerate transformers torch
```

---

## Setup Instructions

### Model Parameters
- **Model Config**: `NousResearch/Nous-Hermes-llama-2-7b`
- **Layers**: 6
- **Dimensions**: 768
- **Heads**: 6
- **Intermediate Size**: 1024
- **Context Length**: 256

### API Keys (Replace with your own):
- **Hugging Face Token**: `hf_************`
- **WandB Token**: `************`

Ensure you replace any sensitive credentials in the code or environment variables.

---

## Training Instructions
1. **Login to Hugging Face**:
   ```python
   from huggingface_hub import login
   login(token="hf_************")
   ```

2. **Login to WandB**:
   ```python
   import wandb
   wandb.login(key="************")
   ```

3. **Dataset Setup**:
   Load the dataset from Hugging Face Hub:
   ```python
   data = load_dataset("abideen/Cosmopedia-100k-pretrain")
   ```

4. **Tokenization**:
   Tokenize the dataset using a customized function to handle truncation and chunking.

5. **Model Creation**:
   Create and configure the Llama model:
   ```python
   from transformers import LlamaForCausalLM
   model = LlamaForCausalLM(config)
   ```

6. **Quantization**:
   Convert the model to a 1-BIT LLM using custom quantization functions:
   ```python
   convert_to_bitnet(model, copy_weights=False)
   ```

7. **Training**:
   Set up the training configuration using `TrainingArguments` and train the model with `Trainer`:
   ```python
   from transformers import Trainer
   trainer = Trainer(
       model=model,
       tokenizer=tokenizer,
       args=args,
       data_collator=data_collator,
       train_dataset=tokenized_data["train"],
   )
   trainer.train()
   ```

---

## Notes
- **Quantization**: The activation and weight quantization significantly reduce model size and computational requirements.
- **Model Output**: The final model is stored in the `./Llama2-70M` directory.
- **Sensitive Information**: Ensure API keys and tokens are managed securely and not shared publicly.

---

## Future Improvements
- Experiment with larger context lengths and layer counts.
- Improve quantization efficiency and accuracy.
- Add validation and testing pipelines.

---

Feel free to contribute, open issues, or suggest enhancements for this project!

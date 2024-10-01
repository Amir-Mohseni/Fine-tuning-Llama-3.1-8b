# LLaMA-3.1-8B-Persian-Instruct

This model is a fine-tuned version of the `meta-llama/Meta-Llama-3.1-8B-Instruct` model, specifically tailored for generating and understanding Persian text. The fine-tuning was conducted using the [TinyStories-Farsi](https://huggingface.co/datasets/taesiri/TinyStories-Farsi) dataset, which includes a diverse set of short stories in Persian. The primary goal of this fine-tuning was to enhance the model's performance in instruction-following tasks within the Persian language.

## Model Details

### Model Description

This model is a fine-tuned version of Llama-3.1-8B-Instruct that meta has released. By training this model on persian short stories, the new model gets to understand the relation between English and Persian in a more meaning full way. 

- **Developed by:** Meta AI 
- **Model type:** Language Model   
- **License:** Apache 2.0  
- **Base Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct`  

### Model Sources

- **Repository:** [Llama-3.1-8B-Instruct on Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

## Training Details

### Training Data
The model was fine-tuned using the [TinyStories-Farsi](https://huggingface.co/datasets/taesiri/TinyStories-Farsi) dataset. This dataset provided a rich and diverse linguistic context, helping the model better understand and generate text in Persian.

### Training Procedure
The fine-tuning process was conducted using the following setup:

- **Epochs:** 4
- **Batch Size:** 8
- **Gradient Accumulation Steps:** 2
- **Hardware:** NVIDIA A100 GPU

### Fine-Tuning Strategy

To make the fine-tuning process efficient and effective, PEFT (Parameter-Efficient Fine-Tuning) techniques were employed. Specifically, the `BitsAndBytesConfig(load_in_4bit=True)` configuration was used, allowing the model to be fine-tuned in 4-bit precision. This approach significantly reduced the computational resources required while maintaining high performance, resulting in a training time of approximately 2 hours. The use of `BitsAndBytesConfig(load_in_4bit=True)` helped reduce the environmental impact by minimizing the computational resources required.

## Uses

### Direct Use

This model is well-suited for generating text in Persian, particularly for instruction-following tasks. It can be used in applications like chatbots, customer support systems, educational tools, and more where accurate and context-aware Persian language generation is needed.

### Out-of-Scope Use

The model is not intended for tasks requiring deep reasoning, complex multi-turn conversations, or contexts beyond the immediate prompt. It is also not designed for generating text in languages other than Persian.

## How to Get Started with the Model

Here is how you can use this model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the combined model
model_name = "AmirMohseni/Llama-3.1-8B-Instruct-Persian-finetuned-sft"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad_token is set (if not already set)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Check if CUDA is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Example usage
input_text = "چطوری میتونم به اطلاعات درباره ی سهام شرکت های آمریکایی دست پیدا کنم؟"

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

# Generate text
outputs = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=512,
    pad_token_id=tokenizer.pad_token_id
)

# Decode and print the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

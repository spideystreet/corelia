---
language: fr
license: apache-2.0
tags:
- medical
- french
- healthcare
- corelia
- mistral
- lora
---

# {model_name}

## Model Description

This model is a LoRA fine-tuned version of Mistral-7B for French medical text processing, developed as part of the Corelia project.

## Training Data

The model was fine-tuned on the following French medical datasets:
- NACHOS (50% weight)
- MediQAl (25% weight) 
- FRASIMED (15% weight)

## Fine-tuning Method

- **Base Model**: mistralai/Mistral-7B-v0.1
- **Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Rank**: 16
- **Alpha**: 32

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{model_name}")
model = AutoModelForCausalLM.from_pretrained("{model_name}")

# Example usage
text = "Patient présentant des symptômes de..."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Performance

This model is designed for French medical text processing and should be evaluated on relevant medical tasks.

## Limitations

- This model is a proof-of-concept and should not be used for medical administrative purposes
- Performance may vary on different medical domains
- Requires proper evaluation before production use

## Citation

```bibtex
@software{{corelia2025,
  title={{Corelia: French Medical AI for Healthcare}},
  author={{Hicham}},
  year={{2025}},
  url={{https://github.com/spideystreet/corelia}}
}}
```

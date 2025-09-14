# Technical Architecture

## Our Research Pipeline

Our approach follows a straightforward but powerful pipeline:

```
Open Sources → Web Crawling → Preprocessing → Mistral-7B → Corelia
   (medical data)   (targeted)   (cleaning)    (continual pre-training)  (French BioMistral)
```

## Building the French Medical Corpus

**Data Collection**: We leverage multiple open-source French medical data sources, including government agencies, medical institutions, and research papers. To ensure comprehensive coverage, we supplement this with targeted web crawling across French medical websites.

**Processing Pipeline**: The collected data undergoes rigorous cleaning and normalization, focusing on French medical terminology standardization and quality filtering to remove irrelevant content.

## Training Strategy

We use [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) as our base model, leveraging its strong French language capabilities. Through continual pre-training on our assembled French medical corpus, we adapt the model to understand medical terminology, healthcare system specifics, and clinical reasoning patterns unique to French medicine.

**Current Focus**: Domain adaptation through continual pre-training. Task-specific fine-tuning will be explored in future research phases.

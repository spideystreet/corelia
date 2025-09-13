# Training Datasets

## Overview

Corelia uses different datasets for each model in our pipeline: French medical datasets for Mistral-7B enhancement and NER datasets for DrBERT fine-tuning.

## Mistral-7B Enhancement Datasets

### 1. NACHOS
- **Source**: [Hugging Face](https://huggingface.co/datasets/chapin/NACHOS_large)
- **License**: Research use
- **Content**: French medical vocabulary and knowledge (~50% of training)

### 2. MediQAl
- **Source**: [Hugging Face](https://huggingface.co/datasets/Abirate/mediqal)
- **License**: Research use
- **Content**: French medical question-answering (~25% of training)

### 3. FRASIMED
- **Source**: [Hugging Face](https://huggingface.co/datasets/alicelacaille/FRASIMED)
- **License**: Research use
- **Content**: French medical clinical cases (~15% of training)

## DrBERT NER Datasets

### 1. MantraGSC
- **Source**: [Hugging Face](https://huggingface.co/datasets/bigbio/mantra_gsc)
- **License**: GPL-3.0
- **Content**: French medical NER annotations

### 2. QUAERO
- **Source**: [Hugging Face](https://huggingface.co/datasets/DrBenchmark/QUAERO)
- **License**: Research use
- **Content**: French medical entity recognition

### 3. MedicalNER_Fr
- **Source**: [Hugging Face](https://huggingface.co/datasets/TypicaAI/MedicalNER_Fr)
- **License**: CC BY 4.0
- **Content**: French medical named entity recognition

## License Considerations

⚠️ **Important**: MantraGSC is GPL-3.0 licensed, requiring our code and models to be open-source.

**License Summary**:
- **MantraGSC**: GPL-3.0 (open-source required)
- **QUAERO**: Research use
- **MedicalNER_Fr**: CC BY 4.0 (attribution required)

## Integration Strategy

### Merging Process
1. **Entity Mapping**: Align entity types across datasets
2. **Quality Control**: Remove duplicates and inconsistencies
3. **Standardization**: Normalize annotation formats
4. **Validation**: Cross-validate annotations for accuracy

### Training Approach
- **Base Model**: DrBERT-7GB pre-trained weights
- **Fine-tuning**: Merged French medical NER dataset
- **Validation**: Held-out test sets for evaluation
- **Quality Assurance**: Medical expert validation and performance metrics

## Data Privacy

### Anonymization
- **Patient Data**: All personal information removed
- **De-identification**: Medical entities preserved, identities protected
- **Compliance**: GDPR and French data protection laws

### Ethical Framework
- **Transparency**: Clear documentation of data sources
- **Attribution**: Proper credit to dataset publishers
- **Bias Mitigation**: Regular assessment and correction

# Training Datasets

## Overview

The first goal of the Corelia project is to create a comprehensive dataset for fine-tuning DrBERT for Named Entity Recognition (NER) extraction. We will merge multiple open-source French medical NER datasets to create a robust training corpus.

## Primary Datasets

### 1. MantraGSC
- **Source**: [Hugging Face Dataset](https://huggingface.co/datasets/bigbio/mantra_gsc)
- **License**: GPL-3.0
- **Content**: French medical NER annotations
- **Size**: Large-scale medical entity recognition corpus

### 2. QUAERO
- **Source**: [Hugging Face Dataset](https://huggingface.co/datasets/DrBenchmark/QUAERO)
- **License**: Research use (check original repository for details)
- **Content**: French medical entity recognition
- **Size**: Comprehensive medical terminology dataset

### 3. MedicalNER_Fr
- **Source**: [Hugging Face Dataset](https://huggingface.co/datasets/TypicaAI/MedicalNER_Fr)
- **License**: CC BY 4.0
- **Content**: French medical named entity recognition
- **Size**: Specialized medical entity annotations

## License Considerations

### GPL-3.0 Compliance
⚠️ **Important**: MantraGSC is published under the GPL-3.0 license, which requires any derived datasets to be distributed under the same license.

**Implications**:
- Our code and trained models must be open-source with GPL-3.0 or equivalent license
- Full compliance with open-source requirements
- Transparent development and distribution

### License Summary
- **MantraGSC**: GPL-3.0 (requires open-source distribution)
- **QUAERO**: Research use (verify original repository)
- **MedicalNER_Fr**: CC BY 4.0 (attribution required)

## Dataset Integration

### Merging Strategy
1. **Entity Mapping**: Align entity types across datasets
2. **Quality Control**: Remove duplicates and inconsistencies
3. **Standardization**: Normalize annotation formats
4. **Validation**: Cross-validate annotations for accuracy

### Mapping Tables
Refer to mapping tables in [mn-datasets_mapping](https://docs.google.com/spreadsheets/u/0/d/14pzWDQYrsx3QyvgVFJeGnk0qkPF9oxe586hVZ2273ig/edit) for detailed entity alignment.

## Training Approach

### Fine-tuning Strategy
1. **Base Model**: Start with DrBERT-7GB pre-trained weights
2. **Domain Adaptation**: Fine-tune on merged French medical NER dataset
3. **Validation**: Use held-out test sets for evaluation
4. **Iteration**: Continuous improvement based on performance metrics

### Quality Assurance
- **Human-in-the-loop**: Medical expert validation
- **Cross-validation**: Multiple validation sets
- **Performance Metrics**: Precision, recall, F1-score
- **Error Analysis**: Systematic error identification and correction

## Data Privacy and Ethics

### Anonymization
- **Patient Data**: All personal information removed
- **De-identification**: Medical entities preserved, identities protected
- **Compliance**: GDPR and French data protection laws

### Ethical Considerations
- **Transparency**: Clear documentation of data sources
- **Attribution**: Proper credit to dataset publishers
- **Responsible Use**: Medical AI best practices
- **Bias Mitigation**: Regular bias assessment and correction

## Future Expansion

### Additional Datasets
- **Hospital Partnerships**: Real-world clinical data
- **Research Collaborations**: Academic medical datasets
- **International Datasets**: Multilingual medical corpora

### Continuous Learning
- **Active Learning**: Human feedback integration
- **Incremental Updates**: Regular model improvements
- **Performance Monitoring**: Ongoing accuracy assessment

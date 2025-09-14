# Research Insights & Technical Choices

## Why We Chose Mistral-7B Over Specialized Biomedical Models

### The Surprising Performance Paradox

Recent research has revealed something counterintuitive: **specialized biomedical models often perform worse** on real-world clinical tasks than general-purpose models. A 2025 study comparing 12 biomedically adapted models found that **11 out of 12 actually exhibited performance declines** on clinical language understanding tasks.

![Medical Fine-tuning Performance Decrease](../assets/medical-finetuning-dicrease-perf.png)

*Source: [Does Biomedical Training Lead to Better Medical Performance?](https://aclanthology.org/2025.gem-1.5.pdf)*

This finding challenges the conventional wisdom that domain-specific models are always better. It suggests that general models, when properly trained, can maintain their broad reasoning capabilities while adapting to specific domains.

### The French Language Challenge

While **BioMistral-7B** excels in English medical applications, it's fundamentally optimized for English processing. This creates significant challenges for French healthcare:

![BioMistral Logo](../assets/logo-biomistral.png)

French medical terminology follows different conventions, clinical documentation uses specific French medical vocabulary, and regulatory compliance requires native French language processing. Most importantly, cultural context significantly affects medical communication patterns - something that English-trained models simply can't capture.

### Why Mistral-7B is Our Optimal Choice

**Mistral-7B** offers unique advantages for creating a French medical AI:

**Native French Foundation**: Unlike BioMistral, Mistral-7B was trained on French text from the ground up, giving it superior understanding of French medical terminology and cultural context.

**Performance Evidence**: Research shows that general-domain models often outperform specialized biomedical models on complex clinical tasks:

![General vs Biomedical Model Performance](../assets/general-model-vs-biomedical-model.png)

*Source: [Does Biomedical Training Lead to Better Medical Performance?](https://aclanthology.org/2025.gem-1.5.pdf)*

**Regulatory Advantages**: As a French model, Mistral-7B provides natural advantages for EU AI Act compliance, data sovereignty requirements, and transparency expectations in French healthcare settings.

![Mistral Logo](../assets/logo-mistral.png)

*Source: [Mistral Large: The Benefits of a French AI Model](https://anthemcreation.com/en/artificial-intelligence/mistral-large-cat-gpt-functioning-benefits-modele-francais/)*

## Our Research Approach

We're leveraging Mistral-7B's strong French foundation and combining it with the largest French medical corpus ever assembled. Through continual pre-training, we're adapting the model to understand French medical terminology, healthcare system specifics, and clinical reasoning patterns - creating what we believe will be the first truly French medical AI.

This approach balances the model's proven general capabilities with specialized medical knowledge, avoiding the performance degradation often seen in overly specialized models while ensuring deep understanding of French medical context.

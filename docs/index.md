# Welcome to Corelia

**AI-powered medical note processing for French healthcare**

Corelia is an open-source AI platform designed to transform unstructured medical reports into structured, actionable insights. Built specifically for French medical terminology and practices, Corelia aims to become the reference solution for medical AI in France.

## Key Features

Corelia provides comprehensive medical note processing through several core capabilities. The platform enhances medical reports by transforming doctor's notes into clean, AI-exploitable insights with 95%+ accuracy rates. Our entity extraction system uses DrBERT-7GB, trained specifically on French hospital corpus, to identify and extract medical entities with superior performance compared to international alternatives. The system automatically maps these entities to standardized French medical codes (ICD-10-FR), ensuring full compliance with national healthcare standards and reducing manual coding time by 80%. Built and hosted entirely in France on OVH's SecNumCloud certified infrastructure, Corelia maintains full GDPR compliance, HDS certification, ANSSI Visa de Sécurité, and complete data sovereignty.

## Technical Architecture

Our three-stage AI pipeline processes medical notes through a carefully designed workflow. LoRA fine-tuned Mistral-7B generates and structures French medical notes with proper terminology and clinical context, creating the French equivalent of specialized biomedical models. DrBERT-7GB performs French medical entity extraction, leveraging its specialized training on French medical terminology. Finally, XMEN maps the extracted entities to ICD-10-FR codes, ensuring standardized output compatible with French healthcare systems.

## Documentation

Explore our comprehensive documentation to understand Corelia's mission and technical approach. The [Project Overview](overview.md) provides a complete understanding of Corelia's mission and strategic approach. Our [System Architecture](architecture.md) details the technical implementation of our AI pipeline with performance specifications. The [Research Insights](research-insights.md) section explains our technical choices and research findings supporting our approach. The [Training Datasets](datasets.md) section covers the French medical NER datasets and licensing considerations. Our [Security & Compliance](security-compliance.md) documentation ensures full transparency on data protection and regulatory compliance. Finally, our [Competitive Advantage](competitive-advantage.md) explains why Corelia is uniquely positioned for French healthcare.

## Vision

Corelia aims to become the reference building block for French medical AI: free, efficient, evolutive, and faithful to the general interest. The platform transforms every medical report into a sovereign and exploitable resource for better care, prevention, and management.

---

*This documentation is built with [MkDocs](https://www.mkdocs.org) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).*

# Welcome to Corelia

<div style="text-align: center; margin: 2rem 0;">
  <img src="assets/logos/logo-and-name.png" alt="Corelia Logo" style="width: 100%; max-width: none; height: auto;">
</div>

**AI-powered medical note processing for French healthcare**

Corelia is an open-source AI platform designed to transform unstructured medical reports into structured, actionable insights. Built specifically for French medical terminology and practices, Corelia aims to become the reference solution for medical AI in France.

## Key Features

Corelia provides comprehensive medical note processing through several core capabilities. The platform enhances medical reports by transforming doctor's notes into clean, AI-exploitable insights with a goal of >90% accuracy rates. Our entity extraction system uses a fine-tuned Mistral-7B for enhancement and normalization, followed by DrBERT-7GB for Named Entity Recognition (NER), trained specifically on French hospital corpus, to identify and extract medical entities with superior performance compared to international alternatives. The system automatically maps these entities to standardized French medical codes (ICD-10-FR), ensuring full compliance with national healthcare standards with a goal of reducing manual coding time by 50-70%. Built and hosted entirely in France on OVH's SecNumCloud certified infrastructure, Corelia maintains full GDPR compliance, HDS certification, ANSSI Visa de Sécurité, and complete data sovereignty.

## Technical Architecture

Our three-stage AI pipeline processes medical notes through a carefully designed workflow. LoRA fine-tuned Mistral-7B generates and structures French medical notes with proper terminology and clinical context, creating the French equivalent of specialized biomedical models. DrBERT-7GB performs French medical entity extraction, leveraging its specialized training on French medical terminology. Finally, XMEN maps the extracted entities to ICD-10-FR codes, ensuring standardized output compatible with French healthcare systems.

## Documentation

Explore our comprehensive documentation to understand Corelia's mission and technical approach:

- **[Project Overview](overview.md)**: Complete understanding of Corelia's mission and strategic approach
- **[System Architecture](architecture.md)**: Technical implementation of our AI pipeline with performance specifications
- **[Research Insights](research-insights.md)**: Technical choices and research findings supporting our approach
- **[Training Datasets](datasets.md)**: French medical NER datasets and licensing considerations
- **[Security & Compliance](security-compliance.md)**: Full transparency on data protection and regulatory compliance
- **[Competitive Advantage](competitive-advantage.md)**: Why Corelia is uniquely positioned for French healthcare
- **[Changelog](changelog.md)**: Project development history and milestones

## Vision

Corelia aims to become the reference building block for French medical AI: easy, efficient, evolutive, and faithful to the general interest. The platform transforms every medical report into a sovereign and exploitable resource for better care, prevention, and management.

---

*This documentation is built with [MkDocs](https://www.mkdocs.org) and the [Material theme](https://squidfunk.github.io/mkdocs-material/).*

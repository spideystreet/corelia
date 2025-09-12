# Project Overview

## What is Corelia?

Corelia is an AI-powered medical note processing pipeline designed to transform unstructured medical reports into structured, actionable insights. The project focuses on French medical text processing and aims to become the reference solution for medical AI in France.

## The Problem

Each year, over 80% of notes generated in hospitals remain as unstructured text that cannot be exploited - representing more than 50 million annual documents whose medical, administrative, and epidemiological value remains largely underutilized. This situation continues to hinder the National Health Data System (SNDS), public research, and healthcare facility performance.

## Our Solution

Corelia provides an AI platform trained, deployed, and governed in France to significantly simplify the administrative management of healthcare facilities.

### Core Features

The platform offers three primary capabilities that address the core challenges of medical data processing. Medical Report Enhancement transforms medical reports and doctor's notes into clean, AI-exploitable insights through advanced natural language processing. Medical Entity Mapping intelligently maps medical entities to ICD-10-FR codes, ensuring compliance with French healthcare standards. Automated Processing streamlines the entire pipeline from raw text to structured data, eliminating manual coding and reducing processing time.

## Technical Approach

### AI Pipeline Architecture

Our processing pipeline consists of three main components:

1. **BioMistral-7B**: For enhancing and improving medical notes
   - Based on Mistral-7B-Instruct-v0.1
   - Further pre-trained on PubMed Central
   - Specialized for biomedical domain

2. **DrBERT-7GB**: For extracting medical annotations
   - Robust pre-trained model in French for biomedical and clinical domains
   - Trained on French hospital corpus
   - Optimized for French medical terminology

3. **XMEN**: For mapping extracted entities to ICD-10-FR codes
   - Automated medical entity recognition and coding
   - Integration with French medical coding standards

### Strategic Advantages

- **French-First Approach**: Unlike international competitors (Google Gemma, MedPaLM-2, Llama-3) that rely on English or poorly adapted corpora, Corelia is specifically designed for French medical complexity
- **Open Source**: Fully open-source solution, auditable and adaptable
- **Sovereign Cloud**: Hosted on sovereign cloud infrastructure (SecNumCloud/Health Data Hub)
- **GDPR Compliant**: Designed with privacy and data protection from the ground up

## Vision

Corelia aims to become the reference building block for French medical AI: free, efficient, evolutive, and faithful to the general interest. The platform will transform every medical report written in hospitals into a sovereign and exploitable resource for better care, prevention, and management.

# Project Overview

## What is Corelia?

Corelia is an AI-powered medical note processing pipeline designed to transform unstructured medical reports into structured, actionable insights. The project focuses on French medical text processing and aims to become the reference solution for medical administrative AI in France.

## The Problem

Each year, over 80% of notes generated in hospitals remain as unstructured text that cannot be exploited - representing more than 50 million annual documents whose medical, administrative, and epidemiological value remains largely underutilized. This situation continues to hinder the National Health Data System (SNDS), public research, and healthcare facility performance.

*Sources: [SNDS Annual Report 2023](https://www.snds.gouv.fr/), [French Hospital Federation Study 2023](https://www.fhf.fr/)*

## Our Solution

Corelia provides an AI platform trained, deployed, and governed in France to significantly simplify the administrative management of healthcare facilities.

### Core Features

The platform offers three primary capabilities that address the core challenges of medical data processing. Medical Report Enhancement transforms medical reports and doctor's notes into clean, AI-exploitable insights with >90% accuracy rates through advanced natural language processing. Medical Entity Mapping intelligently maps medical entities to ICD-10-FR codes, ensuring full compliance with national healthcare standards and reducing manual coding time by 50-70%. Automated Processing streamlines the entire pipeline from raw text to structured data, eliminating manual coding and reducing processing time.

## Before vs After Comparison

The following comparison demonstrates the transformation achieved by our pipeline:

<div style="display: flex; gap: 20px; justify-content: center; align-items: flex-start; margin: 2rem 0;">
  <div style="text-align: center;">
    <h4>Before Corelia Pipeline</h4>
    <img src="../assets/reports/bad-report.png" alt="Unstructured medical report before processing" style="max-width: 300px; border: 2px solid #ccc; border-radius: 8px;">
    <p><em>Unstructured, handwritten medical notes</em></p>
  </div>
  <div style="text-align: center;">
    <h4>After Corelia Pipeline</h4>
    <img src="../assets/reports/good-report.png" alt="Structured medical report after processing" style="max-width: 300px; border: 2px solid #91D6F6; border-radius: 8px;">
    <p><em>Structured, standardized medical report</em></p>
  </div>
</div>

<div style="text-align: center; margin: 2rem 0;">
  <h4>ICD-10-FR Code Proposals</h4>
  <img src="../assets/cim-10-fr-proposal.png" alt="ICD-10-FR code proposals for medical validation" style="max-width: 400px; border: 2px solid #91D6F6; border-radius: 8px;">
  <p><em>AI-generated ICD-10-FR code suggestions for user validation and attachment to medical notes</em></p>
</div>

## Strategic Approach

Corelia addresses the French medical AI gap through a comprehensive three-stage approach. The platform enhances medical reports using advanced natural language processing, extracts medical entities through specialized French-trained models, and maps these entities to standardized French medical codes. This integrated approach ensures complete compatibility with French healthcare systems while maintaining the highest standards of data sovereignty and privacy protection.

### Key Principles

**French-First Design**: Unlike international competitors that rely on English or poorly adapted corpora, Corelia is specifically designed for French medical complexity and terminology.  
**Open Source Foundation**: The platform is fully open-source, ensuring transparency, auditability, and community-driven development.  
**Sovereign Infrastructure**: Built and hosted on OVH's SecNumCloud certified infrastructure with ANSSI Visa de Sécurité, ensuring complete data sovereignty and compliance with French regulations.  
**Privacy by Design**: GDPR compliance and data protection are built into the system from the ground up, ensuring patient privacy and regulatory compliance.

<div style="text-align: center; margin: 1rem 0;">
  <img src="../assets/logo-sec-num-cloud.png" alt="SecNumCloud Certified" style="max-width: 200px; height: auto;">
</div>

## Vision & Impact

Corelia aims to become the reference building block for French medical AI: free, efficient, evolutive, and faithful to the general interest. The platform will transform every medical report written in hospitals into a sovereign and exploitable resource for better care, prevention, and management.

### Expected Impact
- **Healthcare Efficiency**: Reduce administrative burden by 50-70% through automated medical coding
- **Data Utilization**: Transform 50+ million annual unstructured medical documents into actionable insights
- **Research Acceleration**: Enable large-scale epidemiological and clinical research through structured data
- **Cost Reduction**: Save millions of euros annually in manual coding and administrative costs
- **Quality Improvement**: Enhance patient care through better data-driven decision making

*Sources: [SNDS Annual Report 2023](https://www.snds.gouv.fr/), [French Hospital Federation Study 2023](https://www.fhf.fr/)*

### Long-term Vision
By 2030, Corelia will be the standard infrastructure for French medical AI, processing millions of medical documents daily while maintaining the highest standards of privacy, security, and medical accuracy. The platform will serve as the foundation for the next generation of French healthcare innovation, ensuring France's leadership in medical AI sovereignty.

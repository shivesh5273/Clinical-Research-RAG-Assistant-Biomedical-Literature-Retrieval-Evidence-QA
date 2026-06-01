# Clinical Research RAG Assistant — Biomedical Literature Retrieval & Evidence QA

## Project Status

MVP in development.

This project is being built as a domain-specific Retrieval-Augmented Generation system for biomedical and clinical research literature.

The goal is to create a research-literature assistant that retrieves evidence from biomedical papers and generates source-grounded answers while clearly avoiding medical advice, diagnosis, or treatment recommendations.

## Project Overview

The Clinical Research RAG Assistant allows users to ask structured research questions over biomedical or clinical research papers.

The system retrieves relevant passages from uploaded papers and uses those passages as evidence for answer generation. The assistant is designed for literature review, research summarization, and evidence extraction — not for clinical decision-making.

## Important Disclaimer

This project is for biomedical literature retrieval and summarization only.

It does not provide:

- Medical advice
- Diagnosis
- Treatment recommendations
- Clinical decision support
- Patient-specific guidance

All generated outputs should be verified against the original research papers and reviewed by qualified professionals where appropriate.

## Problem Statement

Biomedical research papers are often long, technical, and dense. Researchers and analysts may need to quickly identify:

- Study population
- Intervention
- Comparator
- Outcomes
- Biomarkers
- Risk factors
- Study limitations
- Dataset/cohort information
- Evidence supporting conclusions

General-purpose chatbots may produce unsupported or medically unsafe claims. This project uses RAG to ground answers in retrieved source passages and reduce unsupported generation.

## Objectives

The main objectives of this project are:

1. Build a domain-specific RAG pipeline for biomedical literature.
2. Extract text from clinical or biomedical research PDFs.
3. Split biomedical text into retrievable chunks.
4. Generate embeddings for each chunk.
5. Store embeddings in a vector database.
6. Retrieve relevant passages for research questions.
7. Generate citation-grounded answers.
8. Include a clear research-only safety disclaimer.
9. Evaluate retrieval quality and unsupported answer risks.
10. Document the project clearly for GitHub and portfolio use.

## Key Features

- Biomedical research PDF ingestion
- Text extraction from clinical papers
- Chunking with metadata preservation
- Transformer-based embeddings
- Vector database retrieval
- Research-question answering
- Citation/source snippet display
- Research-only safety disclaimer
- PICO-style question support planned
- Retrieval and hallucination evaluation
- Streamlit interface planned

## Example Questions

The system should be able to answer questions such as:

- What patient population was studied?
- What intervention or treatment was analyzed?
- What comparator was used?
- What were the primary outcomes?
- What were the secondary outcomes?
- What limitations did the authors mention?
- What biomarkers were discussed?
- What risk factors were identified?
- What dataset or cohort was used?
- What evidence supports the conclusion?
- What future research did the authors suggest?

## Planned Tech Stack

- Python
- PyCharm
- Jupyter Notebook
- PyMuPDF
- SentenceTransformers
- FAISS or ChromaDB
- NumPy
- pandas
- scikit-learn
- OpenAI API or local LLM
- Streamlit
- Git and GitHub

Possible future biomedical extensions:

- SciBERT
- BioBERT
- PubMedBERT
- PubMed Central Open Access papers
- ClinicalTrials.gov public data
- PICO extraction templates

## Project Structure

text clinical-research-rag-assistant/ │ ├── README.md ├── requirements.txt ├── app.py │ ├── notebooks/ │   └── 01_clinical_rag_experiments.ipynb │ ├── src/ │   ├── __init__.py │   ├── ingest.py │   ├── chunking.py │   ├── embeddings.py │   ├── retriever.py │   ├── generator.py │   └── evaluation.py │ ├── data/ │   └── sample_papers/ │ ├── reports/ │   └── technical_report.md │ └── screenshots/ 

## RAG Pipeline

The project follows this pipeline:

text Biomedical Research PDFs     ↓ PDF Text Extraction     ↓ Text Cleaning     ↓ Biomedical Text Chunking     ↓ Embedding Generation     ↓ Vector Database Storage     ↓ Research Question     ↓ Semantic Retrieval     ↓ Evidence-Grounded Answer Generation     ↓ Source Snippets + Safety Disclaimer 

## Implementation Plan

### Phase 1: Setup

- Create PyCharm project
- Create virtual environment
- Install required packages
- Create folder structure
- Add sample biomedical research papers

### Phase 2: Biomedical Paper Selection

- Select open-access biomedical or clinical papers
- Use safe public sources
- Avoid patient-specific or private data
- Save PDFs in data/sample_papers/

### Phase 3: PDF Ingestion

- Load PDFs
- Extract text page by page
- Preserve metadata:
  - file name
  - page number
  - chunk ID

### Phase 4: Chunking

- Split biomedical text into meaningful chunks
- Add overlap between chunks
- Preserve source metadata
- Test whether chunk size affects retrieval quality

### Phase 5: Embeddings

- Generate embeddings for each chunk
- Start with a general SentenceTransformers model
- Later compare with biomedical embeddings if time allows

### Phase 6: Vector Search

- Store embeddings using FAISS or ChromaDB
- Convert user questions into embeddings
- Retrieve top-k relevant chunks

### Phase 7: Answer Generation

- Use retrieved chunks as context
- Generate answer using only retrieved evidence
- Display source snippets
- Include research-only disclaimer

### Phase 8: Evaluation

- Create test questions
- Check if retrieved chunks contain supporting evidence
- Track unsupported or hallucinated claims
- Document retrieval errors and limitations

### Phase 9: Interface

- Build a simple Streamlit app
- Allow users to upload papers
- Allow users to ask research questions
- Display answer, sources, and disclaimer

## PICO-Style Extension

A future version may support PICO-style extraction:

text P = Population I = Intervention C = Comparator O = Outcome 

Example output format:

text Population: Intervention: Comparator: Outcome: Evidence: Source: 

This would make the project more useful for clinical research literature review.

## Evaluation Plan

The project will evaluate:

- Whether retrieved chunks contain relevant biomedical evidence
- Whether generated answers are supported by retrieved text
- Whether the system avoids unsupported medical claims
- Whether source snippets match the answer
- Whether general embeddings perform well on biomedical text
- Whether biomedical embeddings improve retrieval quality

Planned evaluation metrics:

- Recall@K
- Mean Reciprocal Rank
- Semantic similarity
- Source-grounding checks
- Hallucination/error analysis

## Current Development Status

- [ ] Project setup
- [ ] Package installation
- [ ] Biomedical paper selection
- [ ] PDF loading
- [ ] Text extraction
- [ ] Chunking
- [ ] Embedding generation
- [ ] Vector search
- [ ] Question-answering
- [ ] Source citation display
- [ ] Disclaimer integration
- [ ] Evaluation
- [ ] Streamlit app
- [ ] GitHub documentation
- [ ] LinkedIn project post

## Future Improvements

- Biomedical embedding model comparison
- PICO extraction
- Clinical-trial document retrieval
- PubMed abstract retrieval
- Hybrid search
- Reranking
- Better citation formatting
- Structured evidence tables
- Research-style technical report
- Streamlit demo interface
- Advanced hallucination detection

## Portfolio Purpose

This project is part of a research-oriented AI/ML portfolio focused on applying RAG to high-value technical domains. It demonstrates domain adaptation, biomedical document retrieval, evidence-grounded answer generation, and responsible handling of sensitive scientific information.

## Disclaimer

This project is for educational and portfolio purposes only. It is not a medical device, clinical decision-support system, or substitute for professional medical judgment.

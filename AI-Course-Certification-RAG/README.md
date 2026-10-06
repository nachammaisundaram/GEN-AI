# AI Course Certification RAG

A document-based Retrieval-Augmented Generation (RAG) application built using Python, LangChain, Google Gemini, and FAISS.

## Overview

This project allows users to ask questions about courses and certifications described in a PDF document.

The application retrieves relevant information from the PDF using vector similarity search and provides the retrieved content to a Gemini language model as context. The model then generates an answer based only on the information available in the document.

If the required information is not available in the document, the application responds:

> I could not find this in the document.

## How It Works

```text
PDF Document
     ↓
PDF Loading
     ↓
Text Chunking
     ↓
Gemini Embeddings
     ↓
FAISS Vector Store
     ↓
Retriever
     ↓
Relevant Context
     ↓
Prompt
     ↓
Gemini LLM
     ↓
Final Answer
```

## Tech Stack

* Python
* LangChain
* Google Gemini
* Gemini Embeddings
* FAISS
* PyPDF
* python-dotenv

## Project Structure

```text
AI-Course-Certification-RAG/
│
├── data/
│   └── courses_certifications.pdf
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd AI-Course-Certification-RAG
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure the Gemini API key

Create a `.env` file in the project root and add your Gemini API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

Do not upload the `.env` file to GitHub.

### 6. Run the application

```bash
python app.py
```

## Example

### Question

> Which certification would be more suitable for someone who wants to enter cloud-focused careers without a programming-heavy start, and why?

### Answer

The Google Cloud Digital Leader certification is suitable because it requires no programming experience and provides foundational cloud knowledge without a programming-heavy start.

## RAG Components

### Document Loading

`PyPDFLoader` loads the certification PDF and converts its pages into documents.

### Text Chunking

`RecursiveCharacterTextSplitter` divides the document into smaller chunks for efficient retrieval.

### Embeddings

Google Gemini embeddings convert the text chunks into vector representations.

### Vector Store

FAISS stores the vectors and performs similarity search.

### Retriever

The retriever finds the most relevant chunks for a user's question.

### Prompt

The prompt instructs the language model to answer using only the retrieved document context.

### LLM

Google Gemini generates the final answer from the retrieved context.

## Future Improvements

* Add an interactive chat interface.
* Support multiple PDF documents.
* Persist the FAISS vector store.
* Add conversation history.
* Improve retrieval for complex multi-part questions.

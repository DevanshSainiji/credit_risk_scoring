# Intelligent Credit Risk Assessment & Agentic Lending System

An end-to-end AI-driven credit risk analytics system that combines Machine Learning (Decision Trees) with **Agentic AI** and **Retrieval-Augmented Generation (RAG)** to provide explainable lending recommendations.

## 1. Project Overview
Traditional credit scoring models often operate as "black boxes." This project evolves the standard ML pipeline into an **Agentic Lending Assistant** that:
1. Predicts the probability of default using a **Decision Tree** model.
2. Retrieves relevant ethical lending guidelines from a local knowledge base using **FAISS**.
3. Generates a structured, human-readable lending report using a **Large Language Model (Llama 3.1)** via LangGraph.
4. Exports the final decision as a professional **PDF report**.

## 2. Key Features
- **Machine Learning**: Decision Tree classifier achieving **94.1% accuracy**.
- **AI Agent**: LangGraph-powered reasoning agent for policy-grounded decisions.
- **RAG Integration**: FAISS vector store with HuggingFace embeddings for guideline retrieval.
- **Interactive UI**: Built with Streamlit for real-time risk assessment.
- **Explainability**: Automated generation of justifications based on retrieved guidelines.
- **PDF Generation**: Instant download of the full assessment report.

## 3. Technology Stack
- **Frontend**: Streamlit
- **ML Models**: Scikit-learn (Decision Tree, Logistic Regression)
- **Agentic Framework**: LangGraph, LangChain
- **LLM**: Llama 3.1 (via Groq Cloud)
- **Vector DB**: FAISS
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **PDF Export**: FPDF2

## 4. Project Structure
```
credit_risk_scoring/
│
├── data/
│   ├── raw/                  # Original dataset (Kaggle)
│   ├── processed/            # Cleaned data with engineered features
│   └── vectorstore/          # FAISS index for RAG
├── src/
│   ├── agent.py              # LangGraph agent definition
│   ├── rag.py                # FAISS retrieval logic
│   ├── pdf_export.py         # PDF generation utility
│   └── ...                   # Preprocessing & Training scripts
├── models/                   # Serialized ML models (.pkl)
├── knowledge_base/           # Ethical lending guidelines (Text)
├── notebooks/                # Exploratory Data Analysis & experiments
├── app.py                    # Main Streamlit application
├── report.tex                # Professional 1-page project report
└── requirements.txt          # Python dependencies
```

## 5. Getting Started

### Prerequisites
- Python 3.8+
- Groq API Key (for Agent functionality)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DevanshSainiji/credit_risk_scoring.git
   cd credit_risk_scoring
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
Launch the Streamlit interface:
```bash
streamlit run app.py
```
*Note: Enter your Groq API Key in the sidebar to enable the Agent Assistant.*

## 6. Team Members
- **Shivansh Bhargava**
- **Devansh Saini**
- **Om Gupta**
- **Sahil Khemnar**

## 7. License
This project is developed for educational purposes as part of the Intelligent Systems curriculum.

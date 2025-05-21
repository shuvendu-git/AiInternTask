# Document Research Chatbot with Flask, LangChain, and Google Gemini AI

This is a Flask web application that allows users to upload text documents, indexes their content using embeddings, and performs AI-powered question answering with citations from the uploaded documents. It uses LangChain, Google Gemini AI (via `langchain_google_genai`), and Chroma vector store for semantic search.

---

## Features

- Upload multiple text files (.txt)
- Automatic document chunking and metadata tagging (document ID, page, paragraph)
- Embedding generation with Google Generative AI Embeddings
- Semantic search with Chroma vector store
- AI question answering with Google Gemini chat model
- Answers include precise citations referencing page and paragraph
- Web interface to upload documents, ask questions, and see cited results

---

## Requirements

- Python 3.8+
- Flask
- langchain
- langchain_community
- langchain_google_genai
- chromadb
- Other dependencies as in `requirements.txt`

---

## 📁 Project Structure
├── app.py
├── requirements.txt
├── templates/
│ └── index.html
├── uploads/
├── file_store.db
├── Dockerfile
└── README.md


## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/shuvendu-git/AiInternTask.git
   cd document-research-chatbot

📄 License
This project is developed as part of an internship task submission. All rights reserved.

🙋 Author
Shuvendu Barik https://github.com/shuvendu-git · https://www.linkedin.com/in/shuvendubarik/
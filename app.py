import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GOOGLE_API_KEY = 'AIzaSyBFsH4z12Ws8ND1wi4I1waDK4AtUcNdbY8'

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
    google_api_key=GOOGLE_API_KEY,
    task_type="retrieval_query"
)

# Setup safety settings - simplified (you can customize or remove)
safety_settings = None

# Initialize chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    safety_settings=safety_settings
)

# Prompt template for QA with citations
prompt_template = """
You are a helpful AI that answers user questions using the given context. Provide precise answers and include citations like 'Page X, Para Y'.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# Global variable for vector DB (in production use persistent DB)
vectordb = None


def load_and_index_documents():
    """
    Load documents from UPLOAD_FOLDER,
    split into chunks with metadata (doc_id, page),
    create or update the vector DB
    """
    global vectordb

    all_texts = []
    doc_files = os.listdir(app.config['UPLOAD_FOLDER'])

    for idx, filename in enumerate(doc_files):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # For simplicity only text files; extend with PDF/OCR later
        if filename.lower().endswith('.txt'):
            loader = TextLoader(filepath, encoding='utf-8')
            documents = loader.load()

            # Add doc_id and fake page metadata (simulate pages by splitting)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = splitter.split_documents(documents)

            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['doc_id'] = filename
                chunk.metadata['page'] = i + 1
                chunk.metadata['paragraph'] = 1  # You can improve paragraph detection

            all_texts.extend(chunks)

    if all_texts:
        vectordb = Chroma.from_documents(all_texts, embedding=embeddings, persist_directory='chroma_db')
        vectordb.persist()
    else:
        vectordb = None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Upload files
        uploaded_files = request.files.getlist('files')
        for file in uploaded_files:
            if file.filename != '':
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(save_path)

        # Reload and reindex documents after upload
        load_and_index_documents()

        return redirect(url_for('index'))

    # List uploaded docs
    uploaded_docs = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', documents=uploaded_docs, results=None)


@app.route('/query', methods=['POST'])
def query():
    question = request.form.get('question', '').strip()
    results = []
    uploaded_docs = os.listdir(app.config['UPLOAD_FOLDER'])

    if not vectordb:
        return render_template('index.html', documents=uploaded_docs, results=results, error="No documents indexed yet. Upload files first.")

    # Setup retriever + QA chain
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=chat_model)

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=multi_retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    response = qa_chain.invoke({"query": question})


    # Parse response and source docs for table
    # source_documents is a list of Document objects with metadata and content
    if 'source_documents' in response:
        for doc in response['source_documents']:
            doc_id = doc.metadata.get('doc_id', 'Unknown')
            page = doc.metadata.get('page', '?')
            para = doc.metadata.get('paragraph', '?')
            citation = f"Page {page}, Para {para}"

            # Get an answer snippet from the chunk (you can improve this to be more precise)
            snippet = doc.page_content.strip().replace('\n', ' ')
            snippet = snippet[:300] + ('...' if len(snippet) > 300 else '')

            results.append({
                'doc_id': doc_id,
                'answer': snippet,
                'citation': citation
            })

    return render_template('index.html', documents=uploaded_docs, results=results, question=question)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # On startup, index any existing docs
    load_and_index_documents()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)


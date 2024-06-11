from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain import hub
import json
from pathlib import Path
from pprint import pprint
from langchain_community.document_loaders import SeleniumURLLoader
from bs4 import BeautifulSoup

from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')



# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_string'
# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf','txt','csv','json','git','html'}
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


store = {}  # This will manage the chat history
llm = ChatOpenAI(model="gpt-4o")
@app.route('/')
def index():
    return render_template('index2.html')

# Global variable
conversational_rag_chain = None
@app.route('/load', methods=['POST'])
def load_document():
    global conversational_rag_chain
    session_id = request.form['session_id']
    loader_type = request.form['loader_type']
    if loader_type == 'url':
        url = request.form['url']
        if not url.startswith('http'):
            return jsonify({'error': 'Invalid URL format'}), 400
        urls = [url]  # Ensure it's a list
        print(f"Received URL: {urls}")
        loader = SeleniumURLLoader(urls=urls, headless=True, browser='chrome', arguments=['--disable-gpu'])
        docs = loader.load()
    elif loader_type == 'pdf':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        loader = PyPDFLoader(filepath)
        docs = loader.load()
    elif loader_type == 'text':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        loader = TextLoader(filepath, 'utf-8')
        docs = loader.load()
    elif loader_type == 'csv':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        loader = CSVLoader(filepath)
        docs = loader.load()
    if not docs:
            return jsonify({'error': 'No documents loaded'}), 400
        
    print(f"Loaded documents: {docs}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    if not splits:
            return 'No document splits created', 400
        
    print(f"Document splits: {splits}")
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    # Initialize and store the RunnableWithMessageHistory
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )
    return ({"message":"Document load successfully"})

@app.route('/ask', methods = ['POST'])
def ask():
    global conversational_rag_chain
    try:
        data = request.get_json()  # This line changes
        question = data['question']
        session_id = data['session_id']

        answer = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )["answer"]
        return jsonify({'answer': answer})
    except KeyError as e:
        return jsonify({'error': f'Missing key: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
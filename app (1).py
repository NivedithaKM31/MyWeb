
from flask import Flask, request, jsonify
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

app = Flask(__name__)

# Load and preprocess documents
PDF_DIRECTORY = "/content/drive/MyDrive/"
MODEL_PATH = "/content/drive/MyDrive/BioMistral-7B.Q4_K_M.gguf"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "your_huggingface_token_here"

def initialize_system():
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    llm = LlamaCpp(model_path=MODEL_PATH, temperature=0.2, max_tokens=2000, top_p=1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    template = """
    <|context|>
    You are a Medical Assistant that follows the instructions and generates accurate responses based on the query and context provided.
    Please be truthful and give direct answers.
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = initialize_system()

@app.route('/process_query', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        response = chain.invoke(query)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

import os
import pickle
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API key from the environment
HF_API_KEY = os.getenv("HF_API_KEY")

# Title of the app
st.title("ReadBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# URL inputs in the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()

# Authenticate Hugging Face API key for embeddings
os.environ["HUGGINGFACE_API_KEY"] = HF_API_KEY  # Set the API key in the environment

# Load HuggingFace embeddings model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Set up the question-answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# When the user clicks 'Process URLs'
if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if url.startswith("http")]
    if not valid_urls:
        st.error("Please enter at least one valid URL.")
    else:
        with st.spinner("Processing URLs and creating embeddings..."):
            # Load data from the URLs
            loader = UnstructuredURLLoader(urls=valid_urls)
            data = loader.load()
            
            # Split data into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            docs = text_splitter.split_documents(data)
            
            # Create vector store using HuggingFace embeddings
            vectorstore_hf = FAISS.from_documents(docs, embeddings)
            
            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_hf, f)
        st.success("Processing complete!")

# Query input field
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        # Load the FAISS index from the pickle file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()
            docs = retriever.get_relevant_documents(query)
            
            # Combine the context from the documents for the question-answering pipeline
            context = " ".join([doc.page_content for doc in docs])
            result = qa_pipeline(question=query, context=context)
            
            # Display the answer
            st.header("Answer")
            st.write(result.get("answer", "No answer available."))

            # Display the sources (URLs) from where the information came
            st.subheader("Sources:")
            for doc in docs:
                st.write(doc.metadata.get("source", "Unknown source"))
    else:
        st.error("FAISS index file not found. Please process URLs first.")


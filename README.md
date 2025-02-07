![Image](https://github.com/user-attachments/assets/7f871604-1e80-45e7-a50e-c4b9167d00ec)
![Image](https://github.com/user-attachments/assets/7fce6cf1-3141-4241-8b5c-bb32b0478061)
# ReadBot: News Research Tool 📈  

ReadBot is a Streamlit-based web application that extracts and processes text from news article URLs. It leverages NLP models to generate embeddings, retrieve relevant information, and answer user queries.  

## Features  
- Extracts text from news article URLs  
- Splits text into smaller chunks for efficient retrieval  
- Uses FAISS for vector-based storage and retrieval  
- Implements Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)  
- Provides answers using `deepset/roberta-base-squad2`  



### Prerequisites  
- Python 3.8+  
- A Hugging Face API key  


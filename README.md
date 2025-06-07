## RAGChatApp
RAGChatApp is a Python-based Retrieval-Augmented Generation (RAG) application that demonstrates document embedding, vector storage, and conversational AI using both Gemini and Llama models. 
The project leverages ChromaDB for vector storage and supports PDF ingestion and querying via chat interfaces.

The goal of this project was to get familiar with the Gemini APIs for closed source models and Ollama APIs for the open source APIs using Llama models. 
I have refrained from using any abstraction libraries like Langchain/LlamaIndex apart from some PDF functionality.

#### Features
* PDF Document Ingestion: Load and split PDF documents into chunks for embedding
* Embeddings: Generate embeddings using either Google Gemini or Sentence Transformers (Llama)
* Vector Store: Store and retrieve document embeddings using ChromaDB
* Conversational Chatbots: Chat interfaces powered by Gemini or Llama models, with document-aware responses
* Web UI: Gradio-based chat interface for user interaction

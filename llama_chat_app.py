import ollama
import gradio as gr
import chromadb
import os
from sentence_transformers import SentenceTransformer


# initialize Chroma DB
persist_directory = "my_local_chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
# get or create a collection
collection = chroma_client.get_or_create_collection(name="llama_rag_pdf_collection")


class Chatbot:
    def __init__(self):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
        self.history = [
            {
                "role": "system",
                "content": """"
                    You are a helpful assistant. Be as helpful as possible, 
                    but do not answer any questions in less than 100 words or more than 200 words.
                    Do not guess the answer if you do not know it.
                    Information from the document takes precedence over your own knowledge.
                """,
            },
        ]

    # Function to search
    def search_similar_chunks(self, query, top_k=2):
        # Embed the query
        embedding = self.model.encode(query, normalize_embeddings=True)

        # Perform similarity search
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        additional_info = []
        # Fix loop to handle actual number of results
        num_results = min(top_k, len(results['distances'][0]))
        for i in range(num_results):
            print(f"\nResult #{i+1}")
            print(f"Similarity Score: {results['distances'][0][i]:.4f}")
            print(f"Chunk ID: {results['metadatas'][0][i]['chunk_id']}")
            print(f"Source Page: {results['metadatas'][0][i]['page_number']}")
            print(f"Text: {results['documents'][0][i]}")
            # Fix similarity threshold to be more reasonable
            if results["distances"][0][i] < 0.8:
                print("Adding to additional info")
                additional_info.append(results["documents"][0][i])

        return additional_info

    def chat_with_ollama(self, message, history):

        additional_info = self.search_similar_chunks(message)
        if additional_info:
            # Fix context formatting
            context = "\n\n".join(additional_info)
            prompt = f"""
            Answer the question based on the context provided below. 
            Do not mention the context explicitly in your response. 
            If the context does not sufficiently answer the question, fall back to your own knowledge. 
            Do not guess if unsure.

            Context: {context}

            Question: {message}
            """
        else:
            prompt = f"""
            Please answer using your own knowledge. 
            Do not guess if you are unsure.

            Question: {message}
            """

        # Fix history management - add original user message
        self.history.append({"role": "user", "content": message})
        
        # Use prompt for generation but don't add it to history
        temp_history = self.history[:-1] + [{"role": "user", "content": prompt}]
        response = ollama.chat(model="llama3.2:latest", messages=temp_history)

        result = response["message"]["content"]
        
        # Fix history management - add assistant response
        self.history.append({"role": "assistant", "content": result})

        return result


chatbot = Chatbot()
demo = gr.ChatInterface(fn=chatbot.chat_with_ollama, type="messages")

# Launch the app
if __name__ == "__main__":
    demo.launch()
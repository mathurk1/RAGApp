from google import genai
from google.genai import types
import gradio as gr
import chromadb
import os


# initialize Gemini API
client = genai.Client(api_key=os.environ.get("API_KEY"))
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="""
            You are a helpful assistant. Be as helpful as possible, 
            but do not answer any questions in less than 100 words or more than 200 words.
            Do not guess the answer if you do not know it.
            Information from the document takes precedence over your own knowledge.
        """,
    ),
)

# initialize Chroma DB
persist_directory = "my_local_chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
# get or create a collection
collection = chroma_client.get_collection(name="gemini_rag_pdf_collection") 


class Chatbot:
    # Function to search
    def search_similar_chunks(self, query, top_k=2):
        # Embed the query
        embedding_response = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=query,
        )
        query_embedding = embedding_response.embeddings[0].values

        # Perform similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        additional_info = []
        for i in range(top_k):
            print(f"\nResult #{i+1}")
            print(f"Similarity Score: {results['distances'][0][i]:.4f}")
            print(f"Chunk ID: {results['metadatas'][0][i]['chunk_id']}")
            print(f"Source Page: {results['metadatas'][0][i]['page_number']}")
            print(f"Text: {results['documents'][0][i]}")
            if results['distances'][0][i] < 0.5:
                print("Adding to additional info")
                # Add the document to additional_info if the similarity score is below 0.5
                additional_info.append(results['documents'][0][i])

        return additional_info


    def chat_with_gemini(self, message, history):

        additional_info = self.search_similar_chunks(message)

        if additional_info:
            prompt = f"""
            Answer the question based on the context provided below. 
            Do not mention the context explicitly in your response. 
            If the context does not sufficiently answer the question, fall back to your own knowledge. 
            Do not guess if unsure.

            Context: {additional_info}

            Question: {message}
            """
        else:
            prompt = f"""
            No relevant context was found for the question. 
            Please answer using your own knowledge. 
            Do not guess if you are unsure.

            Question: {message}
            """
        print(f"Prompt: {prompt}")
        response = chat.send_message_stream(prompt)
        result = ""
        for chunk in response:
            result += chunk.text
            yield result

chatbot = Chatbot()
demo = gr.ChatInterface(fn=chatbot.chat_with_gemini, type="messages")

# Launch the app
if __name__ == "__main__":
    demo.launch()

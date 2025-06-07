from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# initialize the document
loader = PyPDFLoader("PDFs/RAG_Info.pdf")

# initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
)

# Initialize Chroma DB
persist_directory = "my_local_chroma_db"
chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection(name="llama_rag_pdf_collection")

# load the pages
pages_raw = []
for page in loader.lazy_load():
    pages_raw.append(page)

# initialize the SentenceTransformer model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

print("Loaded the PDF and initialized the model")

# procoess each page - create the embedding and load into the vector store
for page in pages_raw:
    page_number = page.metadata["page"]
    texts = text_splitter.create_documents([page.page_content])
    for idx, text in enumerate(texts):
        chunk_text = text.page_content
        embedding = model.encode(chunk_text, normalize_embeddings=True)
        doc_id = str(uuid.uuid4())
        metadata = {
            "source": "RAG_Info.pdf",
            "page_number": page_number,
            "chunk_index": idx,
            "chunk_id": f"page{page_number}_chunk{idx}",
        }

        collection.add(
            ids=[doc_id],
            documents=[chunk_text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
    print("added new chunk to the vector store")
print("All pages processed and added to the vector store")
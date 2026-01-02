#pip install pypdf,langchain, langchain_community, langchain_text_splitters,langchain_pinecone ,pinecone, sentence-transformers, langchain_groq, groq
from getpass import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load PDF and split into chunks
loader = PyPDFLoader("KhushiS_Resume.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs) 

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Initializing Pinecone 
from pinecone import Pinecone, ServerlessSpec

os.environ["PINECONE_API_KEY"] = "pinecone_key_here" 
pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "resume-index"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "gcp",
                "region": "us-east1"
            }
        }
    )

# create vector store
from langchain_pinecone import PineconeVectorStore
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name,
    
)

print("Vector store created.")
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Initializing Groq LLM 
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key="groq_key_here",
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_retries=2,
    
)


# creating a RAG function
def qna_rag(question):
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

   
    messages = [
        ("system", "You are a helpful resume analysis assistant. Use ONLY the provided context."),
        ("human", f"Context:\n{context}\n\nQuestion:\n{question}")
    ]

    # 3. Generate Answer
    response = llm.invoke(messages)
    return response.content

# Example question

question = "Give education details?"
print(qna_rag(question))
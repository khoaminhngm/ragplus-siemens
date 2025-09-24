import os
from getpass import getpass
from pinecone import Pinecone
import langchain
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAI


# Set up Pinecone API key
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY") or getpass(
   "Enter your Pinecone API key: "
)
 
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)


# Initialize Pinecone index for Llama 2 embeddings
index_name = "llama-text-embed-v2"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "text"}
        }
    )
index = pc.Index(index_name)


# Langchain wrapper for Pinecone Embedding
embeddings = PineconeEmbeddings(model="llama-text-embed-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# Delete all data in db before adding new (in case)
vector_store._index.delete(delete_all=True)

# Add Facts and rules to the vector store
# --- Facts ---
document_1 = Document(
    page_content="Siemens was founded in 1847 by Werner von Siemens in Berlin, Germany.",
    metadata={"source": "facts"},
)

document_2 = Document(
    page_content="Siemens is a global leader in electrification, automation, and digitalization solutions.",
    metadata={"source": "facts"},
)

document_3 = Document(
    page_content="Siemens operates in more than 190 countries worldwide.",
    metadata={"source": "facts"},
)

document_4 = Document(
    page_content="Siemens Energy, a spin-off from Siemens AG, focuses on power generation and transmission.",
    metadata={"source": "facts"},
)

document_5 = Document(
    page_content="The Siemens logo is one of the most recognizable industrial logos in the world.",
    metadata={"source": "facts"},
)

# --- Rules ---
document_6 = Document(
    page_content="If a company provides automation and digitalization solutions, then it supports Industry 4.0.",
    metadata={"source": "rules"},
)

document_7 = Document(
    page_content="If a firm has operations in more than 100 countries, then it is considered a multinational corporation.",
    metadata={"source": "rules"},
)

document_8 = Document(
    page_content="If a business is a leader in electrification, then it contributes to energy transition and grid modernization.",
    metadata={"source": "rules"},
)

document_9 = Document(
    page_content="If a company spins off a specialized division, then that division operates as an independent entity.",
    metadata={"source": "rules"},
)

document_10 = Document(
    page_content="If a brand is widely recognized worldwide, then it has strong global brand equity.",
    metadata={"source": "rules"},
)

documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)


# the prompt
prompt = "Why is Siemens considered an important player in the global energy transition?"


# Similarity Search
relevant_docs = vector_store.similarity_search(
    prompt,
    k = 4,
)

facts = [doc for doc in relevant_docs if doc.metadata.get("source") == "facts"]
rules = [doc for doc in relevant_docs if doc.metadata.get("source") == "rules"]

facts_str = "\n".join([f"- {doc.page_content}" for doc in facts])
rules_str = "\n".join([f"- {doc.page_content}" for doc in rules])


# Gemini LLM
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    # google_api_key=os.getenv("GOOGLE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY") or getpass("Enter your Google API key: ")
    )

# Prompt Template
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
    """Based only on the following retrieved relevant facts:
        {facts}
    
        and rules:
        {rules}

        answer the following question:
        {prompt}
    """
)

message = prompt_template.invoke({
    "facts": facts_str,
    "rules": rules_str,
    "prompt": prompt
    })


# Generate response
response = llm.invoke(message)

# Print outputs
print("Relevant Docs retrieved", relevant_docs, "\n")
print("Facts used:\n", facts_str, "\n")
print("Rules used:\n", rules_str, "\n")
print("Prompt:\n", prompt, "\n")

print("Final Response from LLM:", response)
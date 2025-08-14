import getpass
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import time
load_dotenv(override=True)

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
loader = PyPDFLoader('doc_tam.pdf')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separators=" ")
all_splits = text_splitter.split_documents(docs)
print("Total chunks:", len(all_splits))

all_splits = all_splits[:20]
# # Index chunks
_ = vector_store.add_documents(documents=all_splits)
# for i in range(0, len(all_splits), 10):  # 10 chunks per batch
#     batch = all_splits[i:i+10]
#     vector_store.add_documents(batch)
#     time.sleep(2) 

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "Briefly explain this document?"})

print(f"Context: {result['context']}\n\n")
print(f"Answer: {result['answer']}")
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from prompt_templates import memory_prompt_template
import chromadb
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def create_llm(model_path=config['model_path']['small']):
    llm = LlamaCpp(
    model_path=model_path,  
    n_ctx=8192,  
    n_gpu_layers=-1 
)
    return llm

def create_embedding(embedding_path=config['embedding_path']):
    return HuggingFaceBgeEmbeddings(model_name=embedding_path)

def create_chat_memmory(chat_history):
    return ConversationBufferWindowMemory(memory_key='history', chat_memory=chat_history, k=3)
    
def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

    
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)
    
def load_normal_chain(chat_history):
    return ChatChain(chat_history)

def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chroma_db")

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="pdfs",
        embedding_function=embeddings,
    )

    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return PdfChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm=llm, memory=memory, retriever=vector_db.as_retriever(kwargs={"k": 3}))


class PdfChatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memmory(chat_history)
        self.vector_db = load_vectordb(create_embedding())
        llm = create_llm()
        self.llm_chain = load_retrieval_chain(llm, self.memory, self.vector_db)

    def run(self, user_input):
        print("Pdf chat chain is running...")
        return self.llm_chain.invoke(input=user_input, history=self.memory.chat_memory.messages ,stop=["Human:"])['result']
    

class ChatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memmory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm=llm, chat_prompt=chat_prompt, memory=self.memory)

    def run(self, user_input):
        return self.llm_chain.invoke(input=user_input, history=self.memory.chat_memory.messages, stop=['Human:'])['text']

        

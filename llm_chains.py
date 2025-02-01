from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from prompt_templates import memory_prompt_template, pdf_prompt_template
import chromadb
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def create_llm(model_path=config['model_path']['small']):
    llm = LlamaCpp(
    model_path=model_path,  
    n_ctx=2048,  
    n_gpu_layers=-1 
)
    return llm

def create_embedding(embedding_path=config['embedding_path']):
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings( model_name=embedding_path, model_kwargs=model_kwargs, 
                                 encode_kwargs=encode_kwargs)

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

def load_url_chat_chain(chat_history):
    return UrlChatChain(chat_history)

# In load_retrieval_chain()
def load_retrieval_chain(llm, vector_db, memory):  # Add memory parameter
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # Modified prompt template
    prompt = PromptTemplate(
        template=pdf_prompt_template,
        input_variables=["context", "question"]  # Ensure these match your template
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        input_key="question",  
        chain_type="stuff",  
        chain_type_kwargs={
            "prompt": prompt,
        },
        return_source_documents=True,
        output_key='result'
    )


class PdfChatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memmory(chat_history)
        self.vector_db = load_vectordb(create_embedding())
        llm = create_llm()
        self.llm_chain = load_retrieval_chain(llm, self.vector_db, self.memory)

    def run(self, user_input):
        print("Pdf chat chain is running...")
        # Pass input through memory first
        return self.llm_chain.invoke({"question": user_input, "history": self.memory.chat_memory.messages})['result']  


class UrlChatChain(PdfChatChain):
    def __init__(self, chat_history):
        super().__init__(chat_history)


class ChatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memmory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm=llm, chat_prompt=chat_prompt, memory=self.memory)

    def run(self, user_input):
        return self.llm_chain.invoke(input=user_input, history=self.memory.chat_memory.messages, stop=['Human:'])['text']
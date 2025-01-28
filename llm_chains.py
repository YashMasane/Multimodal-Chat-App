from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from prompt_templates import memory_prompt_template
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def create_llm(model_path=config['model_path']['small'], model_type=config['model_type']):
    llm = CTransformers(model=model_path, model_type=model_type)
    return llm

def create_embedding(embedding_path=config['embedding_path']):
    return HuggingFaceInstructEmbeddings(embedding_path)

def create_chat_memmory(chat_history):
    return ConversationBufferWindowMemory(memory_key='history', chat_memory=chat_history, k=3)
    
def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

    
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)
    
def load_normal_chain(chat_history):
    return ChatChain(chat_history)
    

class ChatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memmory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm=llm, chat_prompt=chat_prompt, memory=self.memory)

    def run(self, user_input):
        return self.llm_chain.invoke(input=user_input, history=self.memory.chat_memory.messages, stop=['Human:'])['text']

        

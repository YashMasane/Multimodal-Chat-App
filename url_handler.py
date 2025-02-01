from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_chains import load_vectordb, create_embedding
from langchain_community.document_loaders import WebBaseLoader

def get_url_text(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents

# def extract_text_from_url(pdf_bytes):
#     pdf_file = pypdfium2.PdfDocument(pdf_bytes)
#     return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range()
#                       for page_number in range(len(pdf_file)))

def get_document_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    return splitter.split_documents(docs)

# def get_document_chunks(text_list):
#     documents = []
#     for text in text_list:
#         for chunk in get_text_chunks(text):
#             documents.append(Document(page_content = chunk))
#     return documents

def add_url_documents_to_db(pdfs_bytes):
    docs = get_url_text(pdfs_bytes)
    documents = get_document_chunks(docs)
    vector_db = load_vectordb(create_embedding())
    vector_db.add_documents(documents)
    print("Documents added to db.")
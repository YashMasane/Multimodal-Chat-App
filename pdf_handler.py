from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chains import load_vectordb, create_embedding
from pypdfium2 import PdfDocument

# def get_pdf_text(pdf_bytes_list):
#     return [extract_text_from_pdf(pdf_bytes.getvalue()) for pdf_bytes in pdf_bytes_list]

def get_pdf_text(pdf_bytes_list):
    """Extracts text from multiple PDFs."""
    text_list = []
    for pdf_bytes in pdf_bytes_list:
        pdf = PdfDocument(pdf_bytes.getvalue())  # Read bytes correctly
        text = "\n".join(page.get_textpage().get_text_range() for page in pdf)
        text_list.append(text)
    return text_list

# def extract_text_from_pdf(pdf_bytes):
#     pdf_file = pypdfium2.PdfDocument(pdf_bytes)
#     return "\n".join(pdf_file.get_page(page_number).get_textpage().get_text_range()
#                       for page_number in range(len(pdf_file)))

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=256, separators=["\n\n", "\. ", "\n", " ", ""])
    return splitter.split_text(text)

def get_document_chunks(pdf_bytes_list):
    """Processes multiple PDFs and returns structured document chunks."""
    text_list = get_pdf_text(pdf_bytes_list)  # Extract text from PDFs
    documents = []
    for text in text_list:
        for chunk in get_text_chunks(text):
            documents.append(Document(page_content=chunk))  # Properly wrap in Document
    return documents

def add_documents_to_db(pdfs_bytes_list):
    documents = get_document_chunks(pdfs_bytes_list)
    vector_db = load_vectordb(create_embedding())
    vector_db.add_documents(documents)
    print("Documents added to db.")
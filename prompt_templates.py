memory_prompt_template = """[INST] You are an AI chatbot having a conversation with a human. Answer his questions.
    Previous conversation: {history}
    Human: {human_input}
    AI: [/INST]"""

pdf_prompt_template = """[INST] Use the context below to answer the question:
    
    Context: {context}

    Question: {question} [/INST]"""

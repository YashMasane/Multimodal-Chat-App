import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain, load_url_chat_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import get_timestamp, save_chat_history_json, load_chat_history_json
from image_handler import handle_image
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from url_handler import add_url_documents_to_db
from PIL import Image
import os
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    if st.session_state.pdf_chat:
        print("loading pdf chat chain")
        return load_pdf_chat_chain(chat_history)
    
    if st.session_state.url_chat:
        print("loading url chat chain")
        return load_url_chat_chain(chat_history)
    
    return load_normal_chain(chat_history=chat_history)

def clear_input():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input()

def index_tracker():
    st.session_state.session_index_tracker = st.session_state.session_key

def toggle_pdf_chat():
    st.session_state.pdf_chat = True

def toggle_url_chat():
    st.session_state.url_chat = True

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == 'new_session':
            st.session_state.new_session_key = get_timestamp()+'.json'
            file_path = os.path.join(config['chat_history_path'], st.session_state.new_session_key)
            save_chat_history_json(chat_history=st.session_state.history, file_path=file_path)
        else:
            file_path = os.path.join(config['chat_history_path'], st.session_state.session_key)
            save_chat_history_json(chat_history=st.session_state.history, file_path=file_path)


def main():
    st.title('PolyForma Chatbot')
    st.sidebar.title('Chat Session')
    chat_container = st.container()
    chat_sessions = ['new_session'] + os.listdir(config['chat_history_path'])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox('Select a chat session', chat_sessions, key='session_key', index=index, on_change=index_tracker)
    st.sidebar.toggle('PDF Session', key='pdf_chat', value=False)
    st.sidebar.toggle('URL Session', key='url_chat', value=False)


    
    if st.session_state.session_key != "new_session":
        file_path = os.path.join(config['chat_history_path'], st.session_state.session_key)
        st.session_state.history = load_chat_history_json(file_path)
    else:
        st.session_state.history = []

    user_input = st.text_input('Type you message here...', key='user_input', on_change=set_send_input)

    voice_recording_column, send_button_column = st.columns(2)

    with voice_recording_column:
            voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
    
    with send_button_column:
            send_button = st.button('Send', key='send_button', on_click=clear_input)

    uploaded_url = st.sidebar.text_input('Enter URL here', on_change=toggle_url_chat)
    uploaded_audio = st.sidebar.file_uploader('Upload an Audio', type=['wav', 'mp3', 'ogg'], key='audio_uploaded')
    uploaded_image = st.sidebar.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
    uploaded_pdf = st.sidebar.file_uploader('Upload a PDF', accept_multiple_files=True, type=['pdf'], on_change=toggle_pdf_chat)

    if uploaded_pdf:
        with st.spinner('Processing pdf...'):
            add_documents_to_db(uploaded_pdf)

    if uploaded_url:
        with st.spinner('Processing URL...'):
            add_url_documents_to_db(uploaded_url)

    
    chat_history = StreamlitChatMessageHistory(key='history')
            
    if uploaded_audio:
        file_transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        llm_chat = load_chain(chat_history)
        llm_chat.run("Sumarized this text", file_transcribed_audio)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording['bytes'])
        print(transcribed_audio)
        llm_chat = load_chain(chat_history)
        llm_chat.run(transcribed_audio)
    

    if send_button or st.session_state.send_input:
        if uploaded_image:
            image = Image.open(uploaded_image)
            with chat_container:
                st.image(image, caption="Uploaded Image")
            with st.spinner('Processing image....'):
                user_message = 'Discribed this image in detail'
                if st.session_state.user_question != "":
                    user_message = st.session_state.user_question
                    st.session_state.user_question = ""
                llm_answer = handle_image(uploaded_image.getvalue(), user_message)
                chat_history.add_user_message(user_message)
                chat_history.add_ai_message(llm_answer) 


        if st.session_state.user_question != "":
            st.chat_message('User').write(st.session_state.user_question)
            llm_chat = load_chain(chat_history)
            llm_response = llm_chat.run(st.session_state.user_question)
            st.chat_message('ai').write(llm_response)

            if st.session_state.pdf_chat:
                chat_history.add_user_message(st.session_state.user_question)
                chat_history.add_ai_message(llm_response)

        st.session_state.send_input = False

    if chat_history.messages != []:
        with chat_container:
            for message in chat_history.messages:
                if message.type == 'human':
                    st.chat_message(message.type).write(message.content)
                else:
                    st.chat_message(message.type).write(message.content)

    save_chat_history()

if __name__ == '__main__':
    main()
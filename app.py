import streamlit as st
from llm_chains import load_normal_chain
from langchain.memory import StreamlitChatMessageHistory
from streamlit_mic_recorder import mic_recorder
from utils import get_timestamp, save_chat_history_json, load_chat_history_json
from audio_handler import transcribe_audio
import os
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    return load_normal_chain(chat_history=chat_history)

def clear_input():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input()

def index_tracker():
    st.session_state.session_index_tracker = st.session_state.session_key

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
    chat_container = st.container()
    st.sidebar.title('Chat Session')
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
    
    if st.session_state.session_key != "new_session":
        file_path = os.path.join(config['chat_history_path'], st.session_state.session_key)
        st.session_state.history = load_chat_history_json(file_path)
    else:
        st.session_state.history = []


    chat_history = StreamlitChatMessageHistory(key='history')
    llm_chat = load_chain(chat_history)

    user_input = st.text_input('Type you message here', key='user_input', on_change=set_send_input)
    voice_recording_column, send_button_column = st.columns(2)

    with voice_recording_column:
            voice_recording=mic_recorder(start_prompt="Start recording",stop_prompt="Stop recording", just_once=True)
    
    with send_button_column:
            send_button = st.button('Send', key='send_button', on_click=clear_input)

    uploaded_audio = st.sidebar.file_uploader('Upload an audio', type=['wav', 'mp3', 'ogg'])

    if uploaded_audio:
        file_transcribed_audio = transcribe_audio(uploaded_audio.getvalue())
        llm_chat.run("Sumarized this text", file_transcribed_audio)

    if voice_recording:
        transcribed_audio = transcribe_audio(voice_recording['bytes'])
        print(transcribed_audio)
        llm_chat.run(transcribed_audio)

    print(voice_recording)
    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            st.chat_message('User').write(st.session_state.user_question)
            llm_response = llm_chat.run(st.session_state.user_question)
            st.chat_message('ai').write(llm_response)

    if chat_history.messages != []:
        with chat_container:
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()

if __name__ == '__main__':
    main()
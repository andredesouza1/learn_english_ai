import streamlit as st
from Conversation_Schema import Conversation_Schema, translate_input
import json
import time

current_time = time.localtime()

# Format the time as a string
time_string = time.strftime("%Y-%m-%d %H:%M:%S", current_time)

st.title("Conversar em Ingles")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "translation" not in st.session_state:
    st.session_state.translation = []

if "convo" not in st.session_state:
    st.session_state.convo = ""

if "person" not in st.session_state:
    st.session_state.person = ""

st.session_state.person = st.selectbox("Nome", options=["Aline", "Andre"])

vocab = st.text_input(
    "Escreve as palavras de vocabulario que voce quer praticar.")
st.session_state_convo = None
start_new_conversation = st.button("Nova Conversa")
if start_new_conversation:
    st.session_state.messages = []
    st.session_state.translation = []
    st.session_state.convo = Conversation_Schema(
        vocab=vocab, session_id=time_string, user_id=st.session_state.person)
    st.session_state.convo.init_conv()


input = st.chat_input()


if input:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display user message in chat message container

    st.chat_message("user").markdown(input)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input})

    response = st.session_state.convo.call_llm(input)
    # Display assistant response in chat message container

    response_json_str = response["text"]

    response_json = json.loads(response_json_str)

    analysis_translate = translate_input(response_json["Analysis"])
    conversation_translate = translate_input(response_json["Conversation"])

    with st.chat_message("assistant"):
        st.markdown(response_json["Analysis"])
    # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response_json["Analysis"]})
    with st.chat_message("assistant", avatar="🏖️"):
        st.markdown(analysis_translate.content)
        st.session_state.messages.append(
            {"role": "assistant", "content": analysis_translate.content})
    with st.chat_message("assistant"):
        st.markdown(response_json["Conversation"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response_json["Conversation"]})

    with st.chat_message("assistant", avatar="🏖️"):
        st.markdown(conversation_translate.content)
        st.session_state.messages.append(
            {"role": "assistant", "content": conversation_translate.content})

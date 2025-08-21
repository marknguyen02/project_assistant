import streamlit as st
from generator import stream_custom_chain

st.set_page_config(
    page_title="Estate Assistant", 
)
st.title("Trợ lý dự án bất động sản thông minh")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):

        st.markdown(message["content"])

if prompt := st.chat_input("Tôi có thể hỗ trợ gì cho bạn?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        response_stream = stream_custom_chain(prompt, history=st.session_state.messages)
        full_response = ""

        for chunk in response_stream:
            content = chunk.content or ""
            full_response += content
            response_box.markdown(full_response + "▌")

        response_box.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
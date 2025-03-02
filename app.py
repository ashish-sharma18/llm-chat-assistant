import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

# App Config
st.set_page_config(page_title="LLM Chat Assistant 🤖", page_icon="🤖")
st.title("LLM Chat Assistant 🤖")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your personal Assistant. How can I assist you?"),
    ]

# Function to get AI response
def get_response(user_query, chat_history):
    template = """
    You are a helpful AI assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7, streaming=True)
    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
        st.write(message.content)

# User Input
if user_query := st.chat_input("Type your message..."):
    # Store user message
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("user"):
        st.write(user_query)

    # Stream AI response
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        for chunk in get_response(user_query, st.session_state.chat_history):
            full_response += chunk
            response_container.markdown(full_response)

    # Store AI response
    st.session_state.chat_history.append(AIMessage(content=full_response))

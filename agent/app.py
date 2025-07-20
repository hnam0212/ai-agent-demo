import os

import streamlit as st
from agent_executor import agent
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()

# Page config
st.set_page_config(
    page_title="Larion Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Larion Document Assistant")
st.markdown("Ask questions about Larion's internal documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about Larion documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response from agent
                response = agent.invoke({
                    "messages": [HumanMessage(content=prompt)]
                }, config={"callbacks": [langfuse_handler]})
                
                # Extract the assistant's response
                assistant_message = response["messages"][-1].content
                
                # Display response
                st.markdown(assistant_message)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": assistant_message
                })
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Type your question in the chat")
    st.markdown("2. The agent will search through Larion documents")
    st.markdown("3. Get answers based on internal knowledge")
    
    st.markdown("---")
    st.markdown("**Examples:**")
    st.markdown("- What is our company policy on...?")
    st.markdown("- How do I submit a request for...?")
    st.markdown("- What are the procedures for...?")
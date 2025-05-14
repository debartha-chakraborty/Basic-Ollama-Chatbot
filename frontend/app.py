import streamlit as st
import requests
import json
from typing import List, Dict
import sseclient

# Configuration
BACKEND_URL = "http://localhost:8000"  # Update if your backend is different

def generate_text_stream(prompt: str, model: str) -> requests.Response:
    """Initiate streaming text generation"""
    payload = {
        "prompt": prompt,
        "model": model,
        "stream": True
    }
    return requests.post(
        f"{BACKEND_URL}/api/v1/generate/stream",
        json=payload,
        stream=True
    )

def main():
    st.set_page_config(
        page_title="Simple Ollama Chat",
        page_icon="🤖"
    )
    
    st.title("💬 Ask me Anything!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                response = generate_text_stream(prompt, "gemma2:2b")  # Using llama2 as default model
                message_placeholder = st.empty()
                full_response = ""
                
                client = sseclient.SSEClient(response)
                for event in client.events():
                    try:
                        data = json.loads(event.data)
                        if "text" in data:
                            full_response += data["text"]
                            message_placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        continue
                
                # Finalize the message (remove cursor)
                message_placeholder.markdown(full_response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

if __name__ == "__main__":
    main()
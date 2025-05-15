import streamlit as st
import requests
import json
from typing import List, Dict

# Configuration
BACKEND_URL = "http://localhost:8000"  # Update if your backend is different

def agentic_generate(prompt: str) -> str:
    """Use agentic generation with tools"""
    payload = {
        "prompt": prompt
    }
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/v1/agentic/generate",
            json=payload
        )
        response.raise_for_status()  # Raise exception for non-200 status codes
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with backend: {str(e)}"
    except json.JSONDecodeError:
        return f"Error parsing response from backend: {response.text[:100]}..."

def main():
    st.set_page_config(
        page_title="Federal Register Research Assistant",
        page_icon="📚"
    )
    
    st.title("📚 Federal Register Research Assistant")
    st.subheader("Ask questions about Federal Register documents")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about Federal Register documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = agentic_generate(prompt)
            
            st.markdown(response_text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text
            })

if __name__ == "__main__":
    main()
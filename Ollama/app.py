import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Set page configuration
st.set_page_config(page_title="Q&A Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
        }
        .chat-bubble {
            background-color: #e6e6e6;
            padding: 12px;
            border-radius: 10px;
            margin: 10px 0;
            max-width: 80%;
        }
        .user-bubble {
            background-color: #DCF8C6;
            align-self: flex-end;
        }
        .bot-bubble {
            background-color: #ffffff;
            align-self: flex-start;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar setup
st.sidebar.image("https://img.icons8.com/clouds/100/chatbot.png", width=80)
st.sidebar.title("💬 Chatbot Settings")
selected_model = st.sidebar.selectbox("Choose Model", ["mistral", "gemma:2b", "Gemma2-9b-It"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question:{question}")
])

# Function to generate response
groq_api_key = os.getenv("GROQ_API_KEY")
def generate_response(question: str):
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Main Title
st.title("🤖 Enhanced Q&A Chatbot")

# Input box
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown("### Ask me anything:")
user_input = st.text_input("Your question here...", key="user_input")

# Handle response
if user_input:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-bubble user-bubble">🧑‍💻 {user_input}</div>', unsafe_allow_html=True)
    try:
        bot_response = generate_response(user_input)
        st.markdown(f'<div class="chat-bubble bot-bubble">🤖 {bot_response}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Type a question above to get started!")
st.markdown('</div>', unsafe_allow_html=True)

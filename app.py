import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# LangSmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Page setup
st.set_page_config(page_title="Gemma Chatbot", page_icon="💬")
st.title("Gemma Chatbot (Ollama + LangChain)")
st.write("Ask me anything and I'll respond using the Gemma 2B model running on Ollama.")

# Input field
user_question = st.text_input("Your question:")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

# LLM setup
llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Response
if user_question:
    with st.spinner("Thinking..."):
        answer = chain.invoke({"question": user_question})
    st.markdown("**Answer:**")
    st.write(answer)

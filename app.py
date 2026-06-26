import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="Gemma Chatbot", page_icon="💬")
st.title("Gemma Chatbot (Groq + LangChain)")
st.write("Ask me anything and I'll respond using the Gemma model hosted on Groq Cloud.")

api_key = st.secrets["GROQ_API_KEY"]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the question asked."),
    ("user", "Question: {question}")
])

llm = ChatGroq(groq_api_key=api_key, model_name="gemma-7b-it")  # Groq hosts Gemma models
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

user_question = st.text_input("Your question:")

if user_question:
    with st.spinner("Thinking..."):
        answer = chain.invoke({"question": user_question})
    st.markdown("**Answer:**")
    st.write(answer)

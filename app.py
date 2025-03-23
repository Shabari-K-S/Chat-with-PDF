import os
import streamlit as st
from PyPDF2 import PdfReader

import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If you don't know the answer, just say, "answer is not available in the context provided", don't make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    
    # Use ChatGoogleGenerativeAI instead of GoogleGenerativeAIEmbeddings
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def main():
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs :books:")

    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "options" not in st.session_state:
        st.session_state.options = "offline"

    with st.sidebar:
        st.write(st.session_state.options)
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    get_vectorstore(text_chunks)

                    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                    vectorstore = FAISS.load_local(
                        "faiss_index",
                        embeddings,
                        allow_dangerous_deserialization=True
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chain = get_conversation_chain()

                    st.success("Processing complete! You can now ask questions.")

    if st.session_state.chain and st.session_state.vectorstore:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input for new user message
        if query := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs = st.session_state.vectorstore.similarity_search(query=query, k=3)
                    response = st.session_state.chain.run(input_documents=docs, question=query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()

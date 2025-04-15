import boto3
import streamlit as st
import os
import uuid

## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    llm=Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock_client,
                model_kwargs={"maxTokenCount": 512})
    return llm

# get_response()
def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']


## main method
def main():
    st.set_page_config(page_title="Chat with Knowledgebase", layout="wide")

    # 🌌 Background image
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://images.pexels.com/photos/3214110/pexels-photo-3214110.jpeg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .stApp > div:first-child {
            background-color: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 12px;
            margin: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }
        #MainMenu, footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 Chat with Knowledgebase")

    # 🧠 Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Load index on first run
    if "index_loaded" not in st.session_state:
        with st.spinner("📦 Loading index..."):
            load_index()
            st.session_state.faiss_index = FAISS.load_local(
                index_name="my_faiss",
                folder_path=folder_path,
                embeddings=bedrock_embeddings,
                allow_dangerous_deserialization=True
            )
            st.success("📚 Index is ready!")
            st.session_state.index_loaded = True

    st.markdown("Ask any question based on your uploaded PDFs, Excels, or CSVs:")

    question = st.text_input("🔍 Your question", key="user_input")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a valid question.")
        else:
            with st.spinner("💬 Thinking..."):
                try:
                    llm = get_llm()
                    answer = get_response(llm, st.session_state.faiss_index, question)
                    st.session_state.history.append((question, answer))
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # 📜 Chat history
    if st.session_state.history:
        st.markdown("### 🕘 Chat History")
        for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Q{i}: {q}", expanded=False):
                st.markdown(f"**Answer:** {a}")

    # 🧼 Optional: Clear chat history button
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.experimental_rerun()


if __name__ == "__main__":
    main()
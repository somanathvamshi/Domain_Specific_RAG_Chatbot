import boto3
import streamlit as st
import os
import uuid


## s3_client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")

## Bedrock
from langchain_aws.embeddings import BedrockEmbeddings

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import UnstructuredExcelLoader

from langchain_community.document_loaders import UnstructuredCSVLoader


bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True

def main():
    st.write("This is Admin Site for Chat with PDF and Excel demo")
    uploaded_files = st.file_uploader(
    "Choose PDF, Excel, or CSV files",
    type=["pdf", "xlsx", "xls", "csv"],
    accept_multiple_files=True
)

    if uploaded_files:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        
        all_pages = []

        for uploaded_file in uploaded_files:
            file_id = get_unique_id()
            file_ext = uploaded_file.name.split('.')[-1].lower()
            saved_file_name = f"/tmp/{file_id}.{file_ext}"
            
            # Save the uploaded file
            with open(saved_file_name, mode="wb") as f:
                f.write(uploaded_file.getvalue())
            
            st.write(f"Uploaded: {uploaded_file.name}")

            # Load and split based on file type
            if file_ext == "pdf":
                loader = PyPDFLoader(saved_file_name)
            elif file_ext in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(saved_file_name)
            elif file_ext == "csv":
                loader = UnstructuredCSVLoader(saved_file_name)
            else:
                st.warning(f"Unsupported file type: {file_ext}")
                continue

            pages = loader.load_and_split()
            all_pages.extend(pages)

        st.write(f"Total Chunks from all files: {len(all_pages)}")

        # Split Text
        splitted_docs = split_text(all_pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        st.write("===================")
        st.write(splitted_docs[0])
        st.write("===================")
        st.write(splitted_docs[1])

        # Create Vector Store
        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.success("🎉 All files processed and uploaded to S3 successfully!")
        else:
            st.error("❌ Error! Please check logs.")

if __name__ == "__main__":
    main()
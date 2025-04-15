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
    st.set_page_config(page_title="Admin Uploader", layout="wide")

    # 🎨 Custom Background CSS
    st.markdown(
    """
    <style>
    /* ✅ Fullscreen background image fix for Streamlit 1.44+ */
    .stApp {
        background: url("https://images.pexels.com/photos/3214110/pexels-photo-3214110.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    /* Optional content styling for better contrast */
    .stApp > div:first-child {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }

    /* Hide Streamlit UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

    
    st.title("📚 Admin Panel – Upload PDFs, Excels, and CSVs")

    with st.sidebar:
        st.header("📌 Instructions")
        st.markdown("""
        - Upload **multiple PDFs**, **Excel (.xlsx/.xls)**, or **CSV** files
        - All files will be combined into a searchable knowledge base
        - Output will be stored in **FAISS** and uploaded to **S3**
        """)

    uploaded_files = st.file_uploader(
        "📁 Upload PDF, Excel, or CSV files",
        type=["pdf", "xlsx", "xls", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        request_id = get_unique_id()
        st.info(f"🔑 Unique Request ID: `{request_id}`", icon="🆔")
        
        all_pages = []
        progress_bar = st.progress(0, text="Starting file processing...")

        for idx, uploaded_file in enumerate(uploaded_files):
            file_id = get_unique_id()
            file_ext = uploaded_file.name.split('.')[-1].lower()
            saved_file_name = f"/tmp/{file_id}.{file_ext}"
            
            with open(saved_file_name, mode="wb") as f:
                f.write(uploaded_file.getvalue())

            if file_ext == "pdf":
                loader = PyPDFLoader(saved_file_name)
                file_icon = "📄"
            elif file_ext in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(saved_file_name)
                file_icon = "📊"
            elif file_ext == "csv":
                loader = UnstructuredCSVLoader(saved_file_name)
                file_icon = "🧾"
            else:
                st.warning(f"❗ Unsupported file type: {uploaded_file.name}")
                continue

            st.success(f"{file_icon} Uploaded: **{uploaded_file.name}**")

            try:
                pages = loader.load_and_split()
                all_pages.extend(pages)
                progress_bar.progress((idx + 1) / len(uploaded_files), text=f"Processed {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to load {uploaded_file.name}: {e}")

        st.divider()
        st.subheader("📑 Chunk Overview")

        st.write(f"✅ Total Documents Chunks: `{len(all_pages)}`")

        # Split Text
        splitted_docs = split_text(all_pages, 1000, 200)

        st.write(f"🧠 Vector-ready Chunks: `{len(splitted_docs)}`")

        with st.expander("🔍 Preview First 2 Chunks"):
            st.code(splitted_docs[0].page_content, language='text')
            st.code(splitted_docs[1].page_content, language='text')

        st.divider()
        st.subheader("💾 Vector Store Creation")

        with st.spinner("Creating FAISS vector store and uploading to S3..."):
            result = create_vector_store(request_id, splitted_docs)

        if result:
            st.success("🎉 All files processed and uploaded to S3 successfully!")
        else:
            st.error("❌ Error! Please check logs.")

# def main():
#     st.write("This is Admin Site for Chat with PDF and Excel demo")
#     uploaded_files = st.file_uploader(
#     "Choose PDF, Excel, or CSV files",
#     type=["pdf", "xlsx", "xls", "csv"],
#     accept_multiple_files=True
# )

#     if uploaded_files:
#         request_id = get_unique_id()
#         st.write(f"Request Id: {request_id}")
        
#         all_pages = []

#         for uploaded_file in uploaded_files:
#             file_id = get_unique_id()
#             file_ext = uploaded_file.name.split('.')[-1].lower()
#             saved_file_name = f"/tmp/{file_id}.{file_ext}"
            
#             # Save the uploaded file
#             with open(saved_file_name, mode="wb") as f:
#                 f.write(uploaded_file.getvalue())
            
#             st.write(f"Uploaded: {uploaded_file.name}")

#             # Load and split based on file type
#             if file_ext == "pdf":
#                 loader = PyPDFLoader(saved_file_name)
#             elif file_ext in ["xlsx", "xls"]:
#                 loader = UnstructuredExcelLoader(saved_file_name)
#             elif file_ext == "csv":
#                 loader = UnstructuredCSVLoader(saved_file_name)
#             else:
#                 st.warning(f"Unsupported file type: {file_ext}")
#                 continue

#             pages = loader.load_and_split()
#             all_pages.extend(pages)

#         st.write(f"Total Chunks from all files: {len(all_pages)}")

#         # Split Text
#         splitted_docs = split_text(all_pages, 1000, 200)
#         st.write(f"Splitted Docs length: {len(splitted_docs)}")
#         st.write("===================")
#         st.write(splitted_docs[0])
#         st.write("===================")
#         st.write(splitted_docs[1])

#         # Create Vector Store
#         st.write("Creating the Vector Store")
#         result = create_vector_store(request_id, splitted_docs)

#         if result:
#             st.success("🎉 All files processed and uploaded to S3 successfully!")
#         else:
#             st.error("❌ Error! Please check logs.")

if __name__ == "__main__":
    main()
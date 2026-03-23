import os 
import time
import streamlit as st 
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Configuration
load_dotenv()

# Streamlit Interface
st.set_page_config(page_title="Nvidia NIM RAG", page_icon="🛡️",layout="wide")
st.title("🚀 RAG Q&A With Nvidia NIM & Langchain")

# Sidebar for project controls
st.sidebar.title("Project Controls")

# Api Key 
api_key_input = st.sidebar.text_input("Enter your NVIDIA API Key", type="password", help="Get your key from build.nvidia.com")
env_key = os.getenv("NVIDIA_API_KEY")

# Determine which key to use
final_api_key = api_key_input if api_key_input else env_key

if not final_api_key:
    st.sidebar.error("❌ No API Key detected. Please enter one or check .env")
else:
    os.environ["NVIDIA_API_KEY"] = final_api_key
    if api_key_input:
        st.sidebar.success("✅ Using User-Provided Key")
    else:
        st.sidebar.info("✅ Using System Environment Key")

# Initialize Model
if final_api_key:

    llm = ChatNVIDIA(model="meta/llama3-70b-instruct")  ##NVIDIA NIM Inferencing

# Feature: Document Embedding Logic
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Building Knowledge Base from PDFs..."):
            # 1. Initialize Embeddings
            st.session_state.embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

            # 2. Load and Split
            st.session_state.loader = PyPDFDirectoryLoader("./data")
            docs = st.session_state.loader.load()

            st.session_state.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = 400, chunk_overlap = 50) ## chunk creation
            intermediate_docs = st.session_state.text_splitter.split_documents(docs)

            final_chunks = []
            for doc in intermediate_docs:
                if len(doc.page_content) > 1600: # 1600 chars is roughly 400-450 tokens
                    doc.page_content = doc.page_content[:1600]
                final_chunks.append(doc)
                
            # 5. Build Vector Store
            st.session_state.vectors = FAISS.from_documents(final_chunks, st.session_state.embeddings)
            st.session_state.final_documents = final_chunks

        st.success(f"✅ Success! Created {len(final_chunks)} safe chunks from your documents.")

# Sidebar Interface Info
st.sidebar.markdown("---")
st.sidebar.write("**LLM:** meta/llama3-70b-instruct")
st.sidebar.write("**Vector Store:** FAISS")

if st.sidebar.button("Initialize Document Embedding"):
    if not final_api_key:
        st.error("Please provide an API key first!")
    else:
        vector_embedding()

# User Input section
prompt1 = st.text_input("Enter Your Question From Documents",placeholder="e.g., What was the uninsured rate in 2022?")

if prompt1:
    if not final_api_key:
        st.error("API Key missing. Cannot process request.")

    elif "vectors" not in st.session_state:
        st.warning("Please initialize the document embeddings from the sidebar.")
    else:
        try:
            # Prompt Template
            # Updated Prompt Template for Llama-3-70b
            prompt = ChatPromptTemplate.from_template(
            """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a precise information extraction agent. 
            Your ONLY task is to answer the user's question using the provided context.
            1. DO NOT say "I'd be happy to help" or use introductory filler.
            2. If the answer is not in the context, say "The provided documents do not contain this information."
            3. Use bullet points for state-level data to make it readable.
            4. Base your answer strictly on the <context> tags below.<|eot_id|>

            <|start_header_id|>user<|end_header_id|>
            <context>
            {context}
            </context>

            Question: {input}<|eot_id|>

            <|start_header_id|>assistant<|end_header_id|>
            """
            )
            # Build Retrieval Chain
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
            with st.spinner("🤖 Thinking..."):
                start = time.perf_counter()
                response = retrieval_chain.invoke({'input': prompt1})  
                end = time.perf_counter()

            # Display Results
            col1, col2 = st.columns([3,1])
            with col1:
                st.subheader("Final Answer:")
                st.write(response['answer'])

            with col2:
                # Metric display for performance tracking
                st.metric("Response Time", f"{round(end - start, 2)}s")
                st.metric("Context Chunks", len(response["context"]))
        
            # Feature: Document Similarity Search
            st.markdown("---")
            with st.expander("🔍 Document Similarity Search (Source Chunks)"):
                #Find the relevant chunks
                for i,doc in enumerate(response["context"]):
                    page_info = f" (Page {doc.metadata.get('page', 'N/A')})"
                    st.markdown(f"**Source Chunk {i+1}{page_info}**")
                    st.write(doc.page_content)
                    st.divider()

        except Exception as e:
            st.error(f"An error occurred: {e}")
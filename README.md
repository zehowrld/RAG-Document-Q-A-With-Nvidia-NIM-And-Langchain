# 🛡️ RAG Document Q\&A with NVIDIA NIM & LangChain

[](https://rag-document-q-a-with-nvidia.streamlit.app/)

An advanced **Retrieval-Augmented Generation (RAG)** system that enables conversational intelligence over complex PDF documents. This project leverages **NVIDIA NIM** (NVIDIA Inference Microservices) for state-of-the-art language modeling and **FAISS** for high-performance vector similarity search.

### 🔗 Live Demo

Check out the live application here: [rag-document-q-a-with-nvidia.streamlit.app](https://rag-document-q-a-with-nvidia.streamlit.app/)

-----

## 🚀 Features

  * **NVIDIA NIM Integration:** Utilizes `meta/llama3-70b-instruct` for highly accurate, context-aware responses.
  * **Optimized Vector Embeddings:** Uses `nvidia/nv-embedqa-e5-v5` with token-based splitting to handle dense PDF data.
  * **Dynamic Document Processing:** Built-in support for multi-page PDFs.
  * **Similarity Search Transparency:** Displays the exact source chunks and page metadata used to generate each answer.
  * **Real-time Performance Metrics:** Tracks inference latency and chunk retrieval counts.

-----

## 🛠️ Tech Stack

  * **LLM:** NVIDIA NIM (`meta/llama3-70b-instruct`)
  * **Embeddings:** NVIDIA NIM (`nvidia/nv-embedqa-e5-v5`)
  * **Orchestration:** LangChain & LangChain-Classic
  * **Vector Database:** FAISS (Facebook AI Similarity Search)
  * **Interface:** Streamlit
  * **Data Handling:** PyPDF, Tiktoken

-----

## 📋 Prerequisites

  * Python 3.8+
  * An **NVIDIA API Key** (Generate one at [NVIDIA Build](https://build.nvidia.com/))
  * PDF documents placed in the `/data` directory.

-----

## ⚙️ Setup & Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/zehowrld/RAG-Document-Q-A-With-Nvidia-NIM-And-Langchain.git
    cd RAG-Document-Q-A-With-Nvidia-NIM-And-Langchain
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:

    ```env
    NVIDIA_API_KEY=nvapi-your-key-here
    ```

-----

## 🖥️ Usage

1.  Launch the Streamlit application:
    ```bash
    streamlit run finalapp.py
    ```
2.  Click **"Initialize Document Embedding"** in the sidebar to build the knowledge base.
3.  Ask questions like:
      * *Differences in the Uninsured Rate in the 25 Most Populous Metropolitan Areas in 2022?*
      * *Changes in Public Coverage by State from 2021 to 2022?*

-----

# RAG-Document-Q&A-With-Nvidia-NIM-And-Langchain

This project is a document question-and-answer system utilizing Nvidia's NIM for advanced language modeling and **FAISS** for vector storage and retrieval. The system allows users to interact with documents through a web interface built with Streamlit, providing accurate responses based on the provided context.

## Features

- **Document Embedding**: Converts documents into vector representations using Nvidia's embeddings.
- **Question Answering**: Uses the model **meta/llama3-70b-instruct** to answer questions based on document content.
- **Document Similarity Search**: Retrieves and displays relevant document chunks in response to user queries.

### Prerequisites

- Python 3.8+
- NVIDIA API Key (add to `.env` file)

### Required Libraries

Ensure you have the following libraries installed. You can use `pip` to install them:

```bash
pip install openai python-dotenv langchain_nvidia_ai_endpoints langchain_community faiss-cpu streamlit pypdf
```

### Environment Setup

1. Create a `.env` file in the root directory of your project.
2. Add your NVIDIA API Key to the `.env` file:

    ```env
    NVIDIA_API_KEY=your_nvidia_api_key_here
    ```

## Usage and Deployment

1. **Run the Streamlit Application:**

    ```bash
    streamlit run finalapp.py
    ```

2. **Access the Application:**
   
   Open your web browser and navigate to `http://localhost:8501` to interact with the application.

### Sample Questions to Try

1. **Differences in the Uninsured Rate in the 25 Most Populous Metropolitan Areas in 2022**
   
2. **Changes in Public Coverage by State from 2021 to 2022**:
   
3. **Changes in Private Health Insurance Coverage by State from 2021 to 2022**

## Code Explanation

- **finalapp.py**: The main application file, which includes:
  - Initialization of Nvidia's NIM model.
  - Definition of functions for document embedding and question answering.
  - Streamlit interface for user interaction.

- **vector_embedding()**: Prepares and stores document embeddings for later retrieval.

- **Streamlit Interface:**
  - Input field for users to submit questions.
  - Button to trigger document embedding.
  - Display of answers and document similarity search results.

## Contributing

Feel free to fork the repository and submit pull requests. For issues or suggestions, please open an issue on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any queries or further information, please contact at [hiteshnegi08@gmail.com].

---


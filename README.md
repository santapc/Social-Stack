# Digital MoSJE - Unified Beneficiary Interface (UBI)

## 1. Overview

This is an AI powered system that bridges the gap between beneficiaries and schemes implemented by the government. It allows users to query a knowledge base of schemes and receive relevant information. The app uses vectorstore retrieval to find the most relevant schemes based on the user's query.

The main application is `app.py`. It uses components like `simple_indexer.py` for indexing and `advanced_retriever.py` for retrieving information. `indexapp.py` is the user interface used to add more documents to the vectorstore.

## 2. Technology Used

*   **Python:** The primary programming language.
*   **Streamlit:** A framework for creating interactive web applications.
*   **Langchain:** A framework for building applications powered by language models.
*   **Hugging Face Embeddings:** Used for generating embeddings for the schemes.
*   **Qdrant:** A vector database used to store and retrieve scheme embeddings.
*   **Groq:** A platform for running language models.
*   **dotenv:** For loading environment variables from a `.env` file.

## 3. Architecture

The application follows a RAG architecture:

1.  **Data Ingestion:** Schemes data is ingested and indexed using `simple_indexer.py` and `indexapp.py`.
2.  **Embedding Generation:** Hugging Face Embeddings are used to generate embeddings for the scheme descriptions.
3.  **Vectorstore Storage:** The embeddings are stored in a Qdrant vector database.
4.  **Retrieval:** When a user submits a query, `advanced_retriever.py` retrieves relevant schemes from the vectorstore using vector similarity search.
5.  **Generation:** The retrieved schemes are passed to a language model (LLM) to generate a response to the user's query.
6.  **Response:** The generated response is displayed to the user.

## 4. Prompt Template System

All major prompts used for LLM interactions are now centralized in `components/prompts.py` as template functions. This approach provides:

- **Maintainability:** All prompt templates are in one place, making it easy to update or add new ones.
- **Consistency:** Prompts are formatted using `.format()` with named placeholders, reducing errors and improving readability.
- **Debuggability:** Prompt construction is logged at the debug level for easier troubleshooting.

### How to Use
- To use a prompt, import the relevant function from `components/prompts.py` (e.g., `get_discovery_prompt_template`).
- Call the function and use `.format()` to insert variables:
  ```python
  from components.prompts import get_discovery_prompt_template
  prompt = get_discovery_prompt_template().format(question=my_question, context=my_context, info=my_info)
  ```
- All prompt construction in `components/advanced_retriever.py` and `components/indexing_agent.py` now uses this system.

### Adding or Modifying Prompts
- Edit or add new prompt functions in `components/prompts.py`.
- Each function includes a docstring describing its purpose and variables.
- Update the relevant code to use the new or modified prompt as needed.

## 4. Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <https://github.com/Himank-Khatri/Digital-MoSJE/>
    cd <Digital-MoSJE/>
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    *   Copy `.env.example` to `.env` and fill in the required values.
    *   Add the following environment variables:
        ```
        QDRANT_LINK=<qdrant_url>
        qdrant_api=<qdrant_api_key>
        # ... other variables as in .env.example ...
        ```
    *   Replace `<qdrant_url>` and `<qdrant_api_key>` with your Qdrant instance URL and API key.
4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 10. Troubleshooting & Cross-Platform Notes

- **Tesseract/Poppler:**
  - On Windows, set the Tesseract and Poppler paths in your `.env` file or update `components/config.py`.
  - On Linux, install via package manager and ensure they are in your PATH.
- **Missing Environment Variables:**
  - If you see errors about missing variables, check your `.env` file and ensure all required keys are present.
- **Docker:**
  - The Dockerfile is set up for Linux. For Windows, build/run natively or use WSL2.

## 5. Configuration

The application can be configured using the Streamlit sidebar:

*   **Schemes:** Select the type of schemes to display (All, Central, State).
*   **Select Ministries/States:** Select the specific ministries or states to filter the schemes.
*   **LLM Model:** Select the language model to use for generating responses.
*   **Advanced settings:**
    *   **Enable Multi Query:** Enable the use of multiple queries to improve retrieval.
    *   **Multi Query N:** The number of queries to generate when multi query is enabled.
    *   **Multi Query Retrieval N:** The number of schemes to retrieve for each query when multi query is enabled.

## 6. Data Ingestion

### simple\_indexer.py

This script is used to index data into the Qdrant vectorstore. It reads data from JSON files, generates embeddings using Hugging Face Embeddings, and stores the embeddings in Qdrant.

*   **process\_json\_files(folder\_path):** Reads JSON files from the specified folder, extracts scheme information, and creates Langchain `Document` objects.
*   **create\_vectorstore():** Creates a Qdrant vectorstore from the processed documents.
*   **insert\_into\_vectorstore(new\_documents):** Inserts new documents into the existing vectorstore, avoiding duplicates.
*   **delete\_vectorstore():** Deletes the entire vectorstore.

### indexapp.py

This Streamlit app is used to add new schemes to the vectorstore through a user interface.

1.  **PDF Upload:** Allows users to upload PDF files containing scheme information.
2.  **Agent Extraction:** Uses an `Agent` to extract scheme details from the uploaded PDF.
3.  **Form Input:** Provides a form for users to manually enter scheme details.
4.  **Data Validation:** Validates the required fields (Scheme ID and Scheme Name).
5.  **Vectorstore Insertion:** Adds the new scheme to the Qdrant vectorstore.

## 7. Query Flow

The query flow is handled by `advanced_retriever.py`:

1.  **User Query:** The user enters a query in the Streamlit app.
2.  **Query Analysis:** The `analyze_query` function analyzes the query to determine if it is a broad discovery query or a focused detailed query. It also rewrites the query to improve retrieval coverage.
3.  **Retrieval:** The `create_retriever` function creates a retriever that retrieves relevant schemes from the vectorstore based on the rewritten queries.
4.  **Response Generation:** The `generate_response_streaming` function generates a response to the user's query using the retrieved schemes and a language model.
5.  **Streaming Response:** The response is streamed to the user in chunks.

## 8. Usage Example

1.  Run the application using `streamlit run app.py`.
2.  Enter a query in the chat input, for example, "What are the schemes for OBC students?".
3.  The app will retrieve relevant schemes from the vectorstore and generate a response.
4.  The response will be displayed in the chat interface.


# User Guide for Digital-MoSJE Public Repository

This guide provides instructions on how to set up and use the Digital-MoSJE application.

## 1. Setup

### Prerequisites

Before running the application, ensure you have the following installed:

*   Python 3.8 or higher
*   pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Himank-Khatri/Digital-MoSJE-Public.git
    cd Digital-MoSJE
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 2. Configuration

### API Keys

This application uses external APIs for its functionality. You need to provide your own API keys for certain services.


Open the `.env` file and add the following lines, replacing `YOUR_CHATGPT_API_KEY` with your actual OpenAI API key and `YOUR_GROQ_API_KEY` with your actual Groq API key.

```
OPENAI_API_KEY = "YOUR_CHATGPT_API_KEY"
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
```

**Note on Qdrant API Key:**
The Qdrant API key is pre-configured for read-only access to the vector database and does not need to be changed by the user.

```
qdrant_link= "https://88def10f-ffc0-4bd0-b02f-092aff022b29.europe-west3-0.gcp.cloud.qdrant.io:6333"
qdrant_api= "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJyIn0.QEtqgZpjM1jDZ0ApzZQyj2HRtpaUqgzR5CBEnaC79l8"
```

Other environment variables that may be present in the `.env` file are for internal configuration and generally do not need to be modified by the user.

## 3. Running the Application

To run the Streamlit application, execute the following command in your terminal from the project's root directory:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## 4. Usage

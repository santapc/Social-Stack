# User Guide for Digital-MoSJE Public Repository

This guide provides instructions on how to set up and use the Digital-MoSJE application.

## 1. Setup

### Prerequisites

Before running the application, ensure you have the following installed:

*   Python 3.8 or higher
*   pip (Python package installer)
*   **Tesseract OCR:** Required for Optical Character Recognition (OCR) on images and PDFs. Download and install it from [Tesseract OCR GitHub](https://tesseract-ocr.github.io/tessdoc/Installation.html).
*   **Poppler:** Required for PDF processing. On Windows, you can download pre-built binaries from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows). On Linux, install via your package manager (e.g., `sudo apt-get install poppler-utils`). Ensure Poppler's `bin` directory is added to your system's PATH.

**Note:** While the application may run without Tesseract OCR and Poppler, functionalities related to Optical Character Recognition (OCR) on images and PDF processing will not work correctly or at all without these prerequisites.

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

**Bhashini API Keys:**
If you plan to use Bhashini functionalities, you will need to register for [Bhashini](https://bhashini.gov.in/ulca) and obtain the necessary API keys. Add the following to your `.env` file:

```
BHASHINI_API_KEY=YOUR_BHASHINI_API_KEY
BHASHINI_USER_ID=YOUR_BHASHINI_USER_ID
BHASHINI_PIPELINE_ID=YOUR_BHASHINI_PIPELINE_ID
BHASHINI_INFERENCE_API_KEY=YOUR_BHASHINI_INFERENCE_API_KEY
callback_url=https://dhruva-api.bhashini.gov.in/services/inference/pipeline
APP_NAME=social_stack_ubi
BHASHINI_AI4BHARAT_PIPELINE_ID="643930aa521a4b1ba0f4c41d"
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

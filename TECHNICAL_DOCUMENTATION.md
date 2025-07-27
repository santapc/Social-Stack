# Digital-MoSJE: Technical Documentation

## 1. Project Overview

**Digital-MoSJE** is a comprehensive platform for managing, processing, and retrieving information about government schemes, with a focus on the Ministry of Social Justice and Empowerment (MoSJE). It provides:
- Data extraction from PDFs and APIs
- Indexing and semantic search using LLMs and vector databases
- Multilingual support (via Bhashini)
- Streamlit-based user and admin interfaces
- Modular, extensible architecture

---

## 2. Project Structure

```
Digital-MoSJE/
├── app.py
├── indexapp.py
├── requirements.txt
├── Dockerfile
├── README.md
├── components/
│   ├── aadhaar_extractor.py
│   ├── advanced_retriever.py
│   ├── bhashini.py
│   ├── bhashini_classfile_test_v1.py
│   ├── config.py
│   ├── indexing_agent.py
│   ├── json_indexer.py
│   ├── llms.py
│   ├── prompts.py
│   ├── scrape_api.py
│   ├── simple_indexer.py
│   └── utils.py
├── assets/
│   ├── all_ministries.bin
│   └── all_states.bin
├── scheme_data/
│   ├── cache/
│   ├── errors/
│   └── search/
└── venv/
```

---

## 3. Key Modules and Their Roles

### app.py
- **Main Streamlit application** for end-users.
- Handles user queries (text/voice), language detection, translation, and response generation using LLMs.
- Integrates with Bhashini for multilingual support and audio processing.
- Provides UI for Aadhaar upload and extraction, scheme search, and chat interface.

### indexapp.py
- **Admin/Indexing interface** (Streamlit) for adding new schemes via PDF or URL.
- Extracts scheme data using LLMs and stores it in a Qdrant vector database.
- Allows manual entry and editing of scheme metadata.

### requirements.txt
- Lists all Python dependencies, including Streamlit, LangChain, Qdrant, HuggingFace, OpenAI, PDF/image processing, and more.

### components/
- **aadhaar_extractor.py**: Extracts Aadhaar details (name, DOB, gender, state, etc.) from images/PDFs using OCR (pytesseract).
- **advanced_retriever.py**: Implements semantic search and retrieval using LLMs, embeddings, and Qdrant. Supports multi-query, query rewriting, and streaming responses.
- **bhashini.py**: Integrates with the Bhashini API for language detection, translation, ASR, and TTS.
- **config.py**: Central configuration for model names, dataset names, and external tool paths.
- **indexing_agent.py**: Extracts structured scheme data from PDFs using LLMs and prompt templates. Handles chunking, OCR fallback, and result aggregation.
- **json_indexer.py**: Indexes scheme JSON files into Qdrant, normalizing metadata and generating readable summaries.
- **llms.py**: Utility functions to instantiate LLMs (Ollama, OpenAI) for use throughout the project.
- **prompts.py**: Centralized prompt templates for LLMs (discovery, detailed, extraction, multi-query, etc.).
- **scrape_api.py**: Fetches and cleans scheme data from public APIs, handles caching, error logging, and data normalization.
- **simple_indexer.py**: Simpler alternative for indexing JSON/text files into Qdrant, with BM25 support.
- **utils.py**: Session state management, environment validation, and utility functions for Streamlit apps.

### assets/
- **all_ministries.bin, all_states.bin**: Pickled lists of valid ministries and states for validation and UI selection.

### scheme_data/
- **cache/**: Cleaned, structured scheme data as JSON (one file per scheme, e.g., `scheme_data_15dsugt.json`).
- **search/**: Raw search API results, including scheme slugs, names, and facets (e.g., `search_data_0.json`).
- **errors/**: Error logs and problematic data for debugging.

---

## 4. Data Flow and Architecture

### Data Ingestion
- **API Scraping**: `scrape_api.py` fetches scheme data from myscheme.gov.in APIs, cleans and normalizes it, and stores it in `scheme_data/cache/`.
- **PDF Extraction**: `indexing_agent.py` and `aadhaar_extractor.py` extract structured data from uploaded PDFs/images using OCR and LLMs.

### Indexing
- **json_indexer.py** and **simple_indexer.py** process JSON files in `scheme_data/cache/`, generate readable summaries, and index them into Qdrant for semantic search.
- Metadata fields are validated and normalized (e.g., ministry/state names).

### Retrieval & Search
- **advanced_retriever.py** provides semantic search using embeddings and LLMs, supporting query rewriting, multi-query, and streaming responses.
- **app.py** connects user queries (text/voice) to the retriever, handles translation, and displays results.

### Multilingual & Voice Support
- **bhashini.py** and related logic in `app.py` handle language detection, translation, ASR (speech-to-text), and TTS (text-to-speech) for a multilingual, voice-enabled experience.

---

## 5. Data Formats

### Scheme Data JSON (cache example)
```json
{
  "scheme_id": "66f2d1ea6d8f86fca119a681",
  "slug": "15dsugt",
  "scheme_name": "15 Days Skill Up-gradation Training",
  "scheme_short_title": "15DSUGT",
  "state": "Andhra Pradesh",
  "level": "State/ UT",
  "nodal_department": "Labour Department",
  "dbt_scheme": false,
  "categories": ["Skills & Employment"],
  "sub_categories": ["Training and Skill Up-gradation", "Career information"],
  "target_beneficiaries": ["Individual"],
  "tags": ["Stipend", "Construction Worker", ...],
  "brief_description": "...",
  "detailed_description": "...",
  "benefits": ["Stipend of ₹7,000/-.", ...],
  "eligibility_criteria": "...",
  "application_process": [
    {"mode": "Offline", "steps": ["Step-1: ...", ...]}
  ],
  "references": [
    {"title": "Guidelines", "url": "..."},
    ...
  ],
  "documents_required": "Aadhaar Card of the Worker. ..."
}
```

### Search Data JSON (search example)
- Contains facets, filters, and a list of scheme summaries with fields like `schemeName`, `level`, `schemeCategory`, `slug`, `briefDescription`, `tags`, and eligibility/age info.

---

## 6. Core Workflows

### User Query (Text/Voice)
1. User enters a query or uploads voice (Streamlit UI in `app.py`).
2. Language is detected; if not English, query is translated to English (Bhashini).
3. Query is processed by the retriever (semantic search, LLMs, Qdrant).
4. Results are translated back to the user's language if needed.
5. Optionally, response is converted to speech (TTS).

### Admin Indexing
1. Admin uploads a PDF or enters a scheme URL (`indexapp.py`).
2. Scheme data is extracted using LLMs and prompt templates if admin uploads a pdf else if it is link then scrape via (`scrape_api.py`).
3. Data is validated, normalized, and indexed into Qdrant.

### Data Scraping & Cleaning
1. `scrape_api.py` fetches all scheme slugs and details from the public API.
2. Cleans, normalizes, and caches data in `scheme_data/cache/`.
3. Handles errors and logs problematic cases in `scheme_data/errors/`.

---

## 7. Key Classes and Functions

- **AadhaarExtractor** (`aadhaar_extractor.py`): OCR-based extraction of Aadhaar details from images/PDFs.
- **SimpleRetriever** (`advanced_retriever.py`): Semantic search, query rewriting, multi-query, and streaming LLM responses.
- **BhashiniVoiceAgent** (`bhashini.py`): Handles all Bhashini API interactions (ASR, TTS, translation, language detection).
- **Agent** (`indexing_agent.py`): Extracts structured scheme data from PDFs using LLMs and prompt templates.
- **JsonIndexer/SimpleIndexer**: Indexes scheme data into Qdrant, normalizing metadata and generating summaries.

---

## 8. Configuration & Environment

- **.env**: Stores API keys, Qdrant URLs, Bhashini credentials, and tool paths.
- **config.py**: Centralizes model and dataset names, and tool paths (Tesseract, Poppler).
- **requirements.txt**: All dependencies, including Streamlit, LangChain, Qdrant, HuggingFace, OpenAI, PDF/image processing, and more.

---

## 9. Extending the Project

- **Add new data sources**: Extend `scrape_api.py` or add new modules for additional APIs.
- **Support new languages**: Update `bhashini.py` and prompt templates in `prompts.py`.
- **Improve search**: Enhance `advanced_retriever.py` or integrate new LLMs in `llms.py`.
- **Add new features**: Create new modules in `components/` and update the main app logic in `app.py`.

---

## 10. Troubleshooting & Logs

- **Logs**: `scheme_fetching_cleaning.log` for data processing issues.
- **Errors**: Files in `scheme_data/errors/` for error reports.
- **Cache**: If data seems outdated, clear or update files in `scheme_data/cache/`.

---

## 11. File/Module Reference Table

| File/Module                  | Purpose/Description                                      |
|------------------------------|---------------------------------------------------------|
| app.py                       | Main user-facing Streamlit app                          |
| indexapp.py                  | Admin/indexing Streamlit app                            |
| requirements.txt             | Python dependencies                                     |
| Dockerfile                   | Docker build instructions                               |
| components/aadhaar_extractor.py | Aadhaar OCR extraction                             |
| components/advanced_retriever.py | Semantic search, LLM retrieval                    |
| components/bhashini.py       | Bhashini API integration (ASR, TTS, translation)        |
| components/config.py         | Central config (models, datasets, tool paths)           |
| components/indexing_agent.py | PDF extraction, LLM-based structuring                   |
| components/json_indexer.py   | Indexes JSON scheme data into Qdrant                    |
| components/llms.py           | LLM instantiation utilities                             |
| components/prompts.py        | Prompt templates for LLMs                               |
| components/scrape_api.py     | API scraping, cleaning, caching                         |
| components/simple_indexer.py | Simple JSON/text indexer, BM25 support                  |
| components/utils.py          | Streamlit session state, env validation                 |
| assets/all_ministries.bin    | Pickled list of ministries                              |
| assets/all_states.bin        | Pickled list of states                                  |
| scheme_data/cache/           | Cleaned, structured scheme data (JSON)                  |
| scheme_data/search/          | Raw search API results (JSON)                           |
| scheme_data/errors/          | Error logs and problematic data                         |

---

## 12. Sample Data Field Reference

### Scheme Data (cache)
- `scheme_id`, `slug`, `scheme_name`, `scheme_short_title`, `state`, `level`, `nodal_department`, `dbt_scheme`, `categories`, `sub_categories`, `target_beneficiaries`, `tags`, `brief_description`, `detailed_description`, `benefits`, `eligibility_criteria`, `application_process`, `references`, `documents_required`

### Search Data (search)
- `schemeName`, `level`, `schemeCategory`, `slug`, `briefDescription`, `tags`, `age`, `beneficiaryState`, `nodalMinistryName`, etc.

---

## 13. Contribution & Support

- **Contributions**: Fork the repository, create a feature branch, and submit a pull request.
- **Issues**: Open an issue on the repository with detailed information.
import os
import streamlit as st
import tempfile
import logging
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.components.indexing_agent import Agent
import pickle
from src.services.llms import get_hpc_llm
from langchain_groq import ChatGroq
from src.config import LLM_MODELS, DATASET_NAMES
from src.utils.scrape_api import extract_slug_from_url, fetch_scheme_data


st.set_page_config(layout='wide', page_title="Indexing Endpoint")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { padding: 20px; }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #0055aa;
    }
    .stTextInput>input, .stTextArea>textarea, .stSelectbox>select {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 8px;
    }
    .stFileUploader { margin-bottom: 20px; }
    .section-header { 
        font-size: 1.2em; 
        font-weight: bold; 
        margin-top: 20px; 
        border-bottom: 1px solid #eee; 
        padding-bottom: 5px; 
    }
    .success-box { 
        background-color: #0e7123; 
        padding: 10px; 
        border-radius: 5px; 
        margin-bottom: 20px; 
    }
    </style>
""", unsafe_allow_html=True)

def pickle_read(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return []

def change_llm():
    """Initialize or reinitialize LLM and Agent when llm_model changes"""
    selected_model = st.session_state.get("llm_model_select", st.session_state.get("llm_model", "llama-3.3-70b-versatile"))
    if st.session_state.get("active_llm_model") != selected_model:
        try:
            if "llama" in selected_model:
                st.session_state.llm = ChatGroq(
                    model=selected_model,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2
                )
            elif selected_model in ["gemma3:27b", "qwq:32b", "mistral-small3.1:24b", "deepseek-r1:32b", "deepseek-r1:8b"]:
                st.session_state.llm = get_hpc_llm(model=selected_model)
            st.session_state.active_llm_model = selected_model
            st.session_state.llm_model = selected_model
            # Reinitialize agent with new LLM
            if "chunk_size" in st.session_state:
                with st.spinner("Reinitializing agent..."):
                    st.session_state.agent = Agent(model=st.session_state.llm, chunk_size=st.session_state.chunk_size)
                    st.session_state.agent_initialized = True
            st.info(f"LLM set to {selected_model}")
        except Exception as e:
            st.error(f"Error initializing LLM or Agent: {e}")
            st.session_state.llm = None
            st.session_state.agent = None
            st.session_state.agent_initialized = False

st.title("Indexing Endpoint")
st.markdown("Add new schemes or documents to the dataset with ease.")

# Env check
if 'QDRANT_LINK' not in os.environ or 'QDRANT_API' not in os.environ:
    st.error("Missing Qdrant configuration. Please check environment variables.")
    st.stop()

# Embeddings
if "embeddings" not in st.session_state:
    with st.spinner("Initializing embeddings..."):
        try:
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        except Exception as e:
            st.error(f"Embedding initialization failed: {e}")
            st.stop()

# Initialize session state defaults
if "llm_model" not in st.session_state:
    st.session_state.llm_model = "llama-3.3-70b-versatile"
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

# Success message
if st.session_state.get("show_success", False):
    st.markdown(f'<div class="success-box">Successfully added scheme to {st.session_state.get("last_collection")}</div>', unsafe_allow_html=True)
    st.session_state.show_success = False

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    collection_name = st.selectbox(
        "Dataset",
        DATASET_NAMES,
        help="Select the dataset to store the scheme.",
        key="collection_name_select"
    )
    llm_model = st.selectbox(
        "LLM Model",
        LLM_MODELS,
        index=LLM_MODELS.index(st.session_state.llm_model) if st.session_state.llm_model in LLM_MODELS else 0,
        help="Choose the LLM for Indexing the pdf",
        key="llm_model_select",
        on_change=change_llm
    )
    chunk_size = st.slider(
        "Processing Chunk Size",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
        help="Higher values process faster but may reduce accuracy.",
        key="chunk_size_select"
    )
    st.session_state.chunk_size = chunk_size

# Ensure LLM and Agent are initialized
if "llm" not in st.session_state or st.session_state.get("active_llm_model") != st.session_state.llm_model:
    change_llm()

if not st.session_state.get("agent_initialized") or st.session_state.get("active_llm_model") != st.session_state.llm_model:
    with st.spinner("Initializing agent..."):
        try:
            st.session_state.agent = Agent(model=st.session_state.llm, chunk_size=st.session_state.chunk_size)
            st.session_state.agent_initialized = True
        except Exception as e:
            st.error(f"Agent initialization failed: {e}")
            st.session_state.agent_initialized = False
            st.stop()

# --- New: Option to choose between PDF and Link ---
with st.container():
    st.markdown('<div class="section-header">Add Scheme by PDF or Link</div>', unsafe_allow_html=True)
    input_method = st.radio(
        "Choose input method:",
        options=["PDF", "Link"],
        horizontal=True,
        key="input_method_radio"
    )

    if input_method == "PDF":
        # PDF Upload Section (existing code, moved here)
        uploaded_pdfs = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs to extract scheme details automatically."
        )

        if uploaded_pdfs and st.session_state.agent:
            if st.button("Process PDFs", key="process_pdfs"):
                progress_bar = st.progress(0)
                for idx, uploaded_pdf in enumerate(uploaded_pdfs):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_pdf.read())
                        tmp_path = tmp_file.name
                    try:
                        with st.spinner(f"Processing {uploaded_pdf.name}..."):
                            result = st.session_state.agent.run(f"Extract scheme details from {tmp_path}")
                            extracted = result.get("output", {})
                            if not extracted or "scheme_id" not in extracted:
                                st.warning(f"Could not extract details from {uploaded_pdf.name}")
                            else:
                                # Validate scheme_type
                                extracted_scheme_type = extracted.get("scheme_type", "Central")
                                st.session_state.scheme_type = extracted_scheme_type if extracted_scheme_type in ["Central", "State"] else "Central"
                                # Validate scheme_category against available options
                                category_options = (
                                    pickle_read('assets/all_ministries.bin') if st.session_state.scheme_type == "Central" else
                                    pickle_read('assets/all_states.bin')
                                )
                                extracted_scheme_category = extracted.get("scheme_category", "")
                                st.session_state.scheme_category = (
                                    extracted_scheme_category if extracted_scheme_category in category_options else
                                    category_options[0] if category_options else ""
                                )
                                # Normalize for comparison
                                normalized_category_options = {cat.lower(): cat for cat in category_options}
                                extracted_lower = extracted_scheme_category.lower()

                                if extracted_scheme_category and extracted_lower in normalized_category_options:
                                    st.session_state.scheme_category = normalized_category_options[extracted_lower]
                                else:
                                    logging.warning(f"Extracted scheme_category '{extracted_scheme_category}' is not valid for scheme_type '{st.session_state.scheme_type}'")
                                    st.session_state.scheme_category = category_options[0] if category_options else ""

                                st.session_state.scheme_id = extracted.get("scheme_id", "")
                                st.session_state.scheme_name = extracted.get("scheme_name", "")
                                st.session_state.source_url = extracted.get("source_url", "")
                                st.session_state.details = extracted.get("details", "")
                                st.session_state.benefits = extracted.get("benefits", "")
                                st.session_state.eligibility = extracted.get("eligibility", "")
                                st.session_state.exclusions = extracted.get("exclusions", "")
                                st.session_state.application_process = extracted.get("application_process", "")
                                st.session_state.documents_required = extracted.get("documents_required", "")
                                st.markdown(f'<div class="success-box">Form auto-filled from {uploaded_pdf.name}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_pdf.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
                    progress_bar.progress((idx + 1) / len(uploaded_pdfs))
                # Trigger rerun to update selectboxes
                st.rerun()
    else:
        # Link input section
        scheme_url = st.text_input(
            "Paste the myscheme.gov.in scheme link:",
            value="",
            placeholder="https://www.myscheme.gov.in/schemes/<slug>",
            key="scheme_url_input"
        )
        if st.button("Fetch Scheme from Link", key="fetch_scheme_link"):
            if not scheme_url.strip():
                st.error("Please enter a valid scheme link.")
            else:
                slug = extract_slug_from_url(scheme_url.strip())
                if not slug:
                    st.error("Could not extract scheme slug from the provided link.")
                else:
                    with st.spinner(f"Fetching scheme details for slug: {slug}..."):
                        scheme_data = fetch_scheme_data(slug)
                        if not scheme_data or "scheme_id" not in scheme_data:
                            st.error("Could not fetch scheme details from the link.")
                        else:
                            # Map fields to session state for form population
                            st.session_state.scheme_id = slug
                            st.session_state.scheme_name = scheme_data.get("scheme_name", "")
                            st.session_state.scheme_type = scheme_data.get("level", "Central") if scheme_data.get("level", "Central") in ["Central", "State"] else "Central"
                            # For category, try ministry/state
                            if st.session_state.scheme_type == "Central":
                                category_options = pickle_read('assets/all_ministries.bin')
                                categories = scheme_data.get("categories", [])
                                cat = next((c for c in categories if c in category_options), category_options[0] if category_options else "")
                                st.session_state.scheme_category = cat
                            else:
                                category_options = pickle_read('assets/all_states.bin')
                                state = scheme_data.get("state", "")
                                st.session_state.scheme_category = state if state in category_options else (category_options[0] if category_options else "")
                            st.session_state.source_url = scheme_url.strip()
                            # Prefer detailed_description, fallback to brief_description
                            details = scheme_data.get("detailed_description") or scheme_data.get("brief_description") or ""
                            # Add references to details if available
                            references = scheme_data.get("references", [])
                            if references:
                                ref_str = "\nReferences:\n" + "\n".join(f"- [{ref.get('title','')}]({ref.get('url','')})" for ref in references)
                                details = f"{details}\n{ref_str}"
                            st.session_state.details = details
                            # Benefits: join list if present, else empty
                            st.session_state.benefits = "\n".join(scheme_data.get("benefits", [])) if isinstance(scheme_data.get("benefits", []), list) else scheme_data.get("benefits", "")
                            # Eligibility
                            st.session_state.eligibility = scheme_data.get("eligibility_criteria", "")
                            # Exclusions: map from scheme_data if present
                            st.session_state.exclusions = scheme_data.get("exclusions", "")
                            # Application process: join steps if present
                            app_proc = scheme_data.get("application_process", [])
                            if isinstance(app_proc, list) and app_proc:
                                st.session_state.application_process = "\n".join(
                                    f"Mode: {ap.get('mode', '')} Steps: {'; '.join(ap.get('steps', []))}" for ap in app_proc
                                )
                            else:
                                st.session_state.application_process = ""
                            # Documents required: not present in API, leave blank
                            st.session_state.documents_required = scheme_data.get("documents_required", "")
                            st.markdown(f'<div class="success-box">Form auto-filled from link</div>', unsafe_allow_html=True)
                            st.rerun()

# Manual Scheme Entry Section
with st.container():
    st.markdown('<div class="section-header">Manual Scheme Entry</div>', unsafe_allow_html=True)

    # Initialize session state for scheme_type and scheme_category
    if "scheme_type" not in st.session_state:
        st.session_state.scheme_type = "Central"
    if "scheme_category" not in st.session_state:
        st.session_state.scheme_category = ""

    # Scheme Type and Category outside the form for real-time updates
    col1, col2 = st.columns(2)
    with col1:
        scheme_type = st.selectbox(
            "Scheme Type",
            options=['Central', 'State'],
            index=0 if st.session_state.scheme_type == "Central" else 1,
            help="Select whether the scheme is Central or State-level.",
            key="scheme_type_select"
        )
    with col2:
        category_options = (
            pickle_read('assets/all_ministries.bin') if scheme_type == "Central" else
            pickle_read('assets/all_states.bin')
        )
        # Validate scheme_category against category_options
        if st.session_state.scheme_type != scheme_type or st.session_state.scheme_category not in category_options:
            st.session_state.scheme_type = scheme_type
            st.session_state.scheme_category = category_options[0] if category_options else ""
        scheme_category = st.selectbox(
            "Ministry/State",
            options=category_options,
            index=category_options.index(st.session_state.scheme_category) if st.session_state.scheme_category in category_options else 0,
            help="Select the associated ministry or state.",
            key="scheme_category_select"
        )
        st.session_state.scheme_category = scheme_category

    # Form for other inputs
    with st.form("scheme_form"):
        # Basic Info
        with st.expander("Basic Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                scheme_id = st.text_input(
                    "Scheme ID*",
                    value=st.session_state.get("scheme_id", ""),
                    placeholder="e.g., SCH12345",
                    help="Unique identifier for the scheme."
                )
                scheme_name = st.text_input(
                    "Scheme Name*",
                    value=st.session_state.get("scheme_name", ""),
                    placeholder="e.g., National Health Scheme",
                    help="Full name of the scheme."
                )

        # Additional Details
        with st.expander("Scheme Details"):
            source_url = st.text_input(
                "Source URL",
                value=st.session_state.get("source_url", ""),
                placeholder="e.g., https://example.com/scheme",
                help="Official URL for scheme details."
            )
            details = st.text_area(
                "Details",
                value=st.session_state.get("details", ""),
                placeholder="Describe the scheme in detail...",
                height=100
            )
            benefits = st.text_area(
                "Benefits",
                value=st.session_state.get("benefits", ""),
                placeholder="List the benefits provided by the scheme...",
                height=100
            )

        # Eligibility and Process
        with st.expander("Eligibility and Application"):
            col1, col2 = st.columns(2)
            with col1:
                eligibility = st.text_area(
                    "Eligibility",
                    value=st.session_state.get("eligibility", ""),
                    placeholder="Who is eligible to apply?",
                    height=100
                )
                exclusions = st.text_area(
                    "Exclusions",
                    value=st.session_state.get("exclusions", ""),
                    placeholder="Who is not eligible?",
                    height=100
                )
            with col2:
                application_process = st.text_area(
                    "Application Process",
                    value=st.session_state.get("application_process", ""),
                    placeholder="Steps to apply for the scheme...",
                    height=100
                )
                documents_required = st.text_area(
                    "Documents Required",
                    value=st.session_state.get("documents_required", ""),
                    placeholder="List required documents...",
                    height=100
                )

        # Submit Button
        st.markdown("*Required fields")
        submit_button = st.form_submit_button("Submit Scheme", use_container_width=True)

    # Submit Logic
    if submit_button:
        if not scheme_id or not scheme_name:
            st.error("Scheme ID and Name are required fields.")
        else:
            with st.spinner("Saving scheme..."):
                try:
                    metadata = {
                        "scheme_id": scheme_id,
                        "name": scheme_name,
                        "type": st.session_state.scheme_type,
                        "category": st.session_state.scheme_category,
                        "source_url": source_url
                    }
                    content_dict = {
                        'scheme_id': scheme_id,
                        'scheme name': scheme_name,
                        'scheme type': st.session_state.scheme_type,
                        'scheme category': st.session_state.scheme_category,
                        'details': details,
                        'benefits': benefits,
                        'eligibility': eligibility,
                        'exclusions': exclusions,
                        'application process': application_process,
                        'documents required': documents_required
                    }
                    content = '\n'.join(f'{k}: {v}' for k, v in content_dict.items() if v)
                    doc = Document(page_content=content, metadata=metadata)
                    vectorstore = QdrantVectorStore.from_existing_collection(
                        embedding=st.session_state.embeddings,
                        collection_name=collection_name,
                        url=os.environ['QDRANT_LINK'],
                        api_key=os.environ['QDRANT_API']
                    )
                    vectorstore.add_documents([doc])
                    st.session_state.show_success = True
                    st.session_state.last_collection = collection_name
                    # Clear session state to reset form
                    for key in ['scheme_id', 'scheme_name', 'scheme_type', 'scheme_category', 'source_url', 'details', 'benefits', 'eligibility', 'exclusions', 'application_process', 'documents_required']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving scheme: {e}")
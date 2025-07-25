import streamlit as st
import tempfile
import base64
import os
import asyncio
from dotenv import load_dotenv
import streamlit_mic_recorder as mic
import logging
import pickle
from components.advanced_retriever import SimpleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from components.llms import get_hpc_llm, get_hpc_llm_openai
from components.aadhaar_extractor import AadhaarExtractor
import datetime
from datetime import date
import re
from pydub import AudioSegment
import io
import time
import hashlib
from components.config import LLM_MODELS,groq_models,gpt_models,offline_models
from components.utils import validate_bhashini_setup, initialize_session_state

try:
    from components.bhashini import BhashiniVoiceAgent
except ImportError:
    st.error("Could not import BhashiniVoiceAgent. Make sure the components folder and bhashini.py exist.")
    st.stop()

st.set_page_config(
    page_title="Social Stack",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

BHASHINI_API_KEY = os.getenv("BHASHINI_API_KEY")
BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID")
BHASHINI_INFERENCE_API_KEY = os.getenv("BHASHINI_INFERENCE_API_KEY")
BHASHINI_MEITY_PIPELINE_ID = os.getenv('BHASHINI_MEITY_PIPELINE_ID')
BHASHINI_AI4BHARAT_PIPELINE_ID = os.getenv('BHASHINI_AI4BHARAT_PIPELINE_ID')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BHASHINI_PIPELINE_IDS = {
    'MEITY': BHASHINI_MEITY_PIPELINE_ID,
    'AI4BHARAT': BHASHINI_AI4BHARAT_PIPELINE_ID,
}

def pickle_read(filename):
    """
    Reads a pickled file and returns its contents.

    Args:
        filename (str): Path to the pickled file.

    Returns:
        object: The deserialized object from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def get_embeddings():
    """
    Retrieves HuggingFace embeddings model, cached for performance.

    Returns:
        HuggingFaceEmbeddings: The embeddings model instance.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def debug_log(msg):
    """
    Logs a debug message to session state and console if debug mode is enabled.

    Args:
        msg (str): The message to log.

    Side Effects:
        Appends the message to st.session_state.debug_logs and logs to console if debug_mode is True.
    """
    if st.session_state.debug_mode:
        st.session_state.debug_logs.append(msg)
        logger.info(f"Debug log: {msg}")

def monitor_api_performance():
    """
    Returns the current Bhashini API performance statistics.

    Returns:
        dict: Statistics including total requests, successful requests, failed requests,
              average response time, and language usage.
    """
    return st.session_state.bhashini_stats

def update_performance_stats(success=True, response_time=0, language=None):
    """
    Updates Bhashini API performance statistics.

    Args:
        success (bool): Whether the API call was successful.
        response_time (float): Time taken for the API call in seconds.
        language (str, optional): Language used in the API call.

    Side Effects:
        Updates st.session_state.bhashini_stats with new statistics.
    """
    stats = st.session_state.bhashini_stats
    stats['total_requests'] += 1
    if success:
        stats['successful_requests'] += 1
    else:
        stats['failed_requests'] += 1
    if stats['total_requests'] > 0:
        stats['avg_response_time'] = (
            (stats['avg_response_time'] * (stats['total_requests'] - 1) + response_time) 
            / stats['total_requests']
        )
    if language:
        stats['language_usage'][language] = stats['language_usage'].get(language, 0) + 1

def test_bhashini_connection():
    """
    Tests the connection to the Bhashini API using a silent audio sample.

    Returns:
        dict: Status and message indicating the health of the Bhashini connection.

    Raises:
        Exception: If the API call fails, returns an error message in the status dict.
    """
    try:
        agent = BhashiniVoiceAgent(
            api_key=BHASHINI_API_KEY,
            user_id=BHASHINI_USER_ID,
            inference_api_key=BHASHINI_INFERENCE_API_KEY
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio = AudioSegment.silent(duration=1000, frame_rate=16000)
            audio = audio.set_channels(1).set_sample_width(2)
            audio.export(tmp.name, format="wav")
            audio_b64 = agent.encode_audio(tmp.name)
        test_payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {
                            "sourceLanguage": "hi"
                        }
                    }
                }
            ],
            "inputData": {
                "audio": [{"audioContent": audio_b64}]
            }
        }
        response = agent.call_pipeline(test_payload)
        os.unlink(tmp.name)
        return {"status": "healthy", "message": "Bhashini connection successful"}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Connection test failed: {error_msg}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Server response: {e.response.status_code} - {e.response.text}")
        return {"status": "unhealthy", "message": error_msg}

def show_bhashini_status():
    """
    Displays the status of the Bhashini API connection in the Streamlit UI.

    Returns:
        bool: True if the connection is healthy or in fallback mode, False otherwise.

    Side Effects:
        Displays status messages in the Streamlit app.
    """
    if 'bhashini_status' not in st.session_state:
        with st.spinner("Testing Bhashini connection..."):
            st.session_state.bhashini_status = test_bhashini_connection()
    status = st.session_state.bhashini_status
    if status["status"] == "healthy":
        st.success(f"{status['message']}")
        return True
    elif status["status"] == "circuit_open":
        st.warning("Service temporarily unavailable")
        st.info("The service is in fallback mode. Basic functionality available.")
        return True
    elif status["status"] == "rate_limited":
        st.warning(f"{status['message']}")
        st.info("Please wait a few minutes before trying again.")
        return False
    elif status["status"] == "unhealthy":
        st.error(f"{status['message']}")
        return False
    else:
        st.warning(f"{status['message']}")
        return False

def detect_language(text):
    """
    Detects the language of the input text using Bhashini API.

    Args:
        text (str): Text to analyze for language detection.

    Returns:
        bool: True if the detected language is English, False otherwise.

    Raises:
        Exception: If language detection fails, logs the error and returns True (assumes English).
    """
    try:
        agent = BhashiniVoiceAgent(
            api_key=BHASHINI_API_KEY,
            user_id=BHASHINI_USER_ID,
            inference_api_key=BHASHINI_INFERENCE_API_KEY
        )
        lang = agent.detect_language(text)
        return lang == "en"
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return True

def translate_to_english(text, source_lang):
    """
    Translates text to English using Bhashini API.

    Args:
        text (str): Text to translate.
        source_lang (str): Source language code (e.g., 'hi' for Hindi).

    Returns:
        str: Translated text in English, or original text if translation fails.

    Raises:
        Exception: If translation fails, logs the error and returns the original text.
    """
    try:
        agent = BhashiniVoiceAgent(
            api_key=BHASHINI_API_KEY,
            user_id=BHASHINI_USER_ID,
            inference_api_key=BHASHINI_INFERENCE_API_KEY
        )
        translation_payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_lang,
                            "targetLanguage": "en"
                        }
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text}]
            }
        }
        translated = agent.call_pipeline(translation_payload)
        result = translated["pipelineResponse"][0]["output"][0]["target"]
        return result
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def preserve_urls_during_translation(text, source_lang, target_lang):
    """
    Translates text while preserving URLs and markdown links.

    Args:
        text (str): Text containing URLs or markdown links to translate.
        source_lang (str): Source language code.
        target_lang (str): Target language code.

    Returns:
        str: Translated text with URLs preserved, or original text if translation fails.

    Raises:
        Exception: If translation fails, logs the error and returns the original text.
    """
    url_pattern = r'\[([^\]]+)\]\(([^)]+)\)|(https?://[^\s\)]+)'
    urls = []
    def replace_url(match):
        if match.group(1) and match.group(2):
            urls.append(('markdown', match.group(1), match.group(2)))
            return f"URL_PLACEHOLDER_{len(urls)-1}"
        else:
            urls.append(('plain', None, match.group(3)))
            return f"URL_PLACEHOLDER_{len(urls)-1}"
    text_with_placeholders = re.sub(url_pattern, replace_url, text)
    try:
        agent = BhashiniVoiceAgent(
            api_key=BHASHINI_API_KEY,
            user_id=BHASHINI_USER_ID,
            inference_api_key=BHASHINI_INFERENCE_API_KEY
        )
        translation_payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": source_lang,
                            "targetLanguage": target_lang
                        }
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text_with_placeholders}]
            }
        }
        translated = agent.call_pipeline(translation_payload)
        translated_text = translated["pipelineResponse"][0]["output"][0]["target"]
        for i, (url_type, text, url) in enumerate(urls):
            placeholder = f"URL_PLACEHOLDER_{i}"
            if url_type == 'markdown':
                translated_text = translated_text.replace(placeholder, f"[{text}]({url})")
                translated_text = re.sub(r'\s*\]\s*\(\s*', '](', translated_text)
                translated_text = re.sub(r'\s*\)\s*', ')', translated_text)
            else:
                translated_text = translated_text.replace(placeholder, url)
        translated_text = re.sub(r'\s*-\s*', '-', translated_text)
        translated_text = re.sub(r'\s*,\s*', ', ', translated_text)
        translated_text = re.sub(r'\s*\.\s*', '. ', translated_text)
        return translated_text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def format_markdown_links(text):
    """
    Formats markdown links to ensure consistent spacing.

    Args:
        text (str): Text containing markdown links.

    Returns:
        str: Text with formatted markdown links.
    """
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    def format_link(match):
        text = match.group(1).strip()
        url = match.group(2).strip()
        return f"[{text}]({url})"
    result = re.sub(link_pattern, format_link, text)
    return result

def translate_preserving_structure(text, source_lang, target_lang):
    parts = []
    current_pos = 0
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    # Split text into parts (text and links)
    for match in re.finditer(link_pattern, text):
        if match.start() > current_pos:
            parts.append(('text', text[current_pos:match.start()]))
        link_text = match.group(1)
        url = match.group(2)
        parts.append(('link', link_text, url))
        current_pos = match.end()

    if current_pos < len(text):
        parts.append(('text', text[current_pos:]))

    translated_parts = []

    agent = BhashiniVoiceAgent(
        api_key=BHASHINI_API_KEY,
        user_id=BHASHINI_USER_ID,
        inference_api_key=BHASHINI_INFERENCE_API_KEY
    )

    for part in parts:
        if part[0] == 'text':
            lines = part[1].split('\n')
            to_translate = []
            line_types = []  # To track if line is a scheme name or detail
            
            for line in lines:
                if line.strip():
                    # Identify scheme names (lines starting with '-' at beginning of line)
                    if line.lstrip() == line and line.startswith('-'):
                        line_types.append('name')
                    else:
                        line_types.append('detail')
                    to_translate.append(line)
                else:
                    line_types.append('empty')

            translated_map = {}
            if to_translate:
                translation_payload = {
                    "pipelineTasks": [
                        {
                            "taskType": "translation",
                            "config": {
                                "language": {
                                    "sourceLanguage": source_lang,
                                    "targetLanguage": target_lang
                                }
                            }
                        }
                    ],
                    "inputData": {
                        "input": [{"source": line} for line in to_translate]
                    }
                }
                translated = agent.call_pipeline(translation_payload)
                results = [item["target"] for item in translated["pipelineResponse"][0]["output"]]
                translated_map = dict(zip(to_translate, results))

            # Rebuild lines with proper formatting
            final_lines = []
            trans_index = 0
            for i, line in enumerate(lines):
                if not line.strip():
                    final_lines.append('')
                    continue
                    
                translated_line = translated_map.get(line, line)
                if line_types[i] == 'name':
                    # Remove bullet point from scheme name and add extra newline
                    final_lines.append('\n' + translated_line.lstrip('-').strip())
                else:
                    # Ensure details start with bullet points
                    if not translated_line.lstrip().startswith('-'):
                        translated_line = '- ' + translated_line.lstrip()
                    final_lines.append(translated_line)
                
                trans_index += 1

            translated_parts.append(('text', '\n'.join(final_lines)))

        elif part[0] == 'link':
            link_text = part[1]
            url = part[2]

            translation_payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "language": {
                                "sourceLanguage": source_lang,
                                "targetLanguage": target_lang
                            }
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": link_text}]
                }
            }
            translated = agent.call_pipeline(translation_payload)
            translated_name = translated["pipelineResponse"][0]["output"][0]["target"]
            translated_parts.append(('link', f"[{translated_name}]({url})"))

    # Combine all parts with proper formatting
    result = []
    for part in translated_parts:
        if part[0] == 'link':
            result.append(part[1])
        else:
            # Add content with proper spacing
            content = part[1].strip()
            if content:
                result.append(content)

    # Join with double newlines to ensure proper spacing
    return '\n\n'.join(result)

def validate_audio(audio_bytes):
    """
    Validates the format of audio bytes for Bhashini processing.

    Args:
        audio_bytes (bytes): Audio data in WAV format.

    Returns:
        bool: True if the audio format is valid, False otherwise.

    Raises:
        ValueError: If audio format is invalid (wrong sample rate, channels, or sample width).
    """
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        logger.info(f"Validated audio: Sample rate={audio.frame_rate}, Channels={audio.channels}, Sample width={audio.sample_width}, Duration={len(audio)/1000}s")
        if audio.frame_rate not in [8000, 16000]:
            raise ValueError(f"Unsupported sample rate: {audio.frame_rate}. Expected 8kHz or 16kHz.")
        if audio.sample_width != 2:
            raise ValueError(f"Unsupported sample width: {audio.sample_width}. Expected 16-bit.")
        if audio.channels != 1:
            raise ValueError(f"Unsupported channels: {audio.channels}. Expected mono.")
        logger.info("Audio format is valid.")
        return True
    except Exception as e:
        logger.error(f"Invalid audio file: {str(e)}")
        return False

def convert_audio_to_required_format(audio_bytes):
    """
    Converts audio to the required format for Bhashini (16-bit, mono, 16kHz WAV).

    Args:
        audio_bytes (bytes): Raw audio data.

    Returns:
        bytes: Converted audio bytes, or None if conversion fails.

    Raises:
        ValueError: If the audio data is invalid or too short.
    """
    try:
        debug_path = os.path.join(tempfile.gettempdir(), f"raw_audio_{int(time.time())}.wav")
        with open(debug_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"Saved raw audio for debugging at: {debug_path}")
        if len(audio_bytes) < 44:
            raise ValueError("Audio data too short to contain a valid WAV header.")
        if not audio_bytes.startswith(b'RIFF'):
            raise ValueError("Invalid WAV file: Missing RIFF header.")
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        logger.info(f"Original audio: Sample rate={audio.frame_rate}, Channels={audio.channels}, Sample width={audio.sample_width}, Duration={len(audio)/1000}s")
        audio = audio.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        output = io.BytesIO()
        audio.export(output, format="wav")
        converted_bytes = output.getvalue()
        logger.info(f"Converted audio size: {len(converted_bytes)} bytes")
        return converted_bytes
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        logger.error(f"Debug audio saved at: {debug_path}")
        return None

async def main():
    """
    Main function to run the Streamlit application for the MOSJE Voice Assistant.

    Handles UI setup, user input processing, LLM interactions, and audio processing.

    Side Effects:
        Initializes session state, sets up UI, processes user inputs, and updates chat history.
    """
    initialize_session_state()

    st.title("Social Stack: Transforming Welfare Delivery")
    # st.markdown("*Ask about government schemes in your preferred Indic language*")
    
    if not validate_bhashini_setup():
        st.error("Please configure your Bhashini credentials to continue.")
        st.stop()
    
    embeddings = get_embeddings()
    
    if "llm" not in st.session_state or st.session_state.llm_model != st.session_state.get("active_llm_model"):
        try:
            if st.session_state.llm_model in groq_models:
                st.session_state.llm = ChatGroq(
                    model=st.session_state.llm_model,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )
            elif st.session_state.llm_model in offline_models:
                st.session_state.llm = get_hpc_llm(model=st.session_state.llm_model)
            elif st.session_state.llm_model in gpt_models:
                st.session_state.llm = get_hpc_llm_openai(model=st.session_state.llm_model)
            st.session_state.active_llm_model = st.session_state.llm_model
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            st.stop()

    if (
        "retriever" not in st.session_state or 
        st.session_state.retriever.collection_name != st.session_state.dataset
    ):
        st.session_state.retriever = SimpleRetriever(
            collection_name=st.session_state.dataset,
            llm=st.session_state.llm_model,
            embeddings=embeddings,
        )
    else:
        st.session_state.retriever.llm = st.session_state.llm

    

    st.sidebar.header("🔧 Configuration")
    dataset_options = ["All", "Central", "State"]
    dataset_type_index = dataset_options.index(st.session_state.dataset_type) if st.session_state.dataset_type in dataset_options else 0
    dataset_type = st.sidebar.radio(
        "Schemes",
        dataset_options,
        index=dataset_type_index,
        key="dataset_type_unique"
    )
    if dataset_type != st.session_state.dataset_type:
        st.session_state.dataset_type = dataset_type
        if dataset_type == "Central":
            st.session_state['central_selection_unique'] = st.session_state.ministries_list.copy()
            st.session_state['state_selection_unique'] = []
        elif dataset_type == "State":
            st.session_state['state_selection_unique'] = st.session_state.states_list.copy() if not st.session_state.get('sync_aadhaar', False) else [st.session_state.aadhaar_state] if st.session_state.aadhaar_state else []
            st.session_state['central_selection_unique'] = []
        else:
            st.session_state['central_selection_unique'] = []
            st.session_state['state_selection_unique'] = []

    with st.sidebar.expander("Select Ministries/States"):
        if dataset_type == 'Central':
            options = st.session_state.ministries_list
            key = 'central_selection_unique'
        elif dataset_type == 'State':
            options = st.session_state.states_list
            key = 'state_selection_unique'
        else:
            options = []
            key = 'all_selection_unique'

        if dataset_type in ['Central', 'State']:
            if key not in st.session_state:
                st.session_state[key] = options.copy()

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Select All", key=f"select_all_{dataset_type.lower()}_unique"):
                    st.session_state[key] = options.copy()
            with col2:
                if st.button("Clear All", key=f"clear_all_{dataset_type.lower()}_unique"):
                    st.session_state[key] = []

            if dataset_type == 'State' and st.session_state.get('sync_aadhaar', False) and st.session_state.aadhaar_state in st.session_state.states_list:
                default_selection = [st.session_state.aadhaar_state]
            else:
                default_selection = st.session_state[key]

            category = st.multiselect(
                "Select " + ("Ministries" if dataset_type == "Central" else "States"),
                options=options,
                default=default_selection,
                key=key
            )
            if category != st.session_state[key]:
                st.session_state[key] = category
                st.session_state['sync_aadhaar'] = False

    if "aadhaar_info_list" in st.session_state and st.session_state.aadhaar_info_list:
        with st.sidebar.expander("Extracted Aadhaar Info", expanded=True):
            for i, info in enumerate(st.session_state.aadhaar_info_list):
                for k, v in info.items():
                    st.markdown(f"- **{k.capitalize()}**: {v}")
            states_extracted = [i["state"] for i in st.session_state.aadhaar_info_list if i.get("state")]
            if states_extracted:
                st.session_state.aadhaar_state = states_extracted[0]
                st.session_state.aadhaar_filter_enabled = True
                if st.sidebar.button("Sync with Aadhaar", key="sync_aadhaar_button_unique"):
                    if st.session_state.aadhaar_state in st.session_state.states_list:
                        st.session_state.dataset_type = "State"
                        st.session_state['state_selection_unique'] = [st.session_state.aadhaar_state]
                        st.session_state['sync_aadhaar'] = True
                    else:
                        st.error("Aadhaar state not found in available states.")
                        st.session_state['sync_aadhaar'] = False
            else:
                st.warning("No state extracted from Aadhaar.")

    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Advanced Settings")
    
    with st.sidebar.expander("🤖 Advanced LLM Settings"):
        llm_model = st.selectbox(
            "LLM Model",
            groq_models + gpt_models,
            index=LLM_MODELS.index(st.session_state.llm_model),
        )
        st.session_state.llm_model = llm_model
        st.session_state.discovery_top_n = st.slider("Default No. of Schemes", min_value=1, max_value=12, value=5)
        st.session_state.conversation_memory_size = st.slider("Conversation Memory Size", min_value=0, max_value=10, value=5)
        # st.session_state.use_multi_query = st.checkbox("Enable Multi Query", value=st.session_state.use_multi_query)
        # if st.session_state.use_multi_query:
        #     st.session_state.multi_query_n = st.number_input("Multi Query N", 1, 10, st.session_state.multi_query_n)
        #     st.session_state.multi_query_ret_n = st.number_input("Multi Query Retrieval N", 1, 10, st.session_state.multi_query_ret_n)

    with st.sidebar.expander("🔧 Advanced Audio Settings"):
        voice_gender = st.selectbox("Voice Gender for TTS", ["female", "male"], index=0)

    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    

    chat_container = st.container()
    chat_placeholder = st.empty()

    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role, avatar="🧠" if role == "assistant" else None):
                st.markdown(message)

    @st.dialog("Upload Aadhaar")
    def aadhaar_upload_dialog():
        """
        Displays a dialog for uploading and processing Aadhaar images or PDFs.

        Side Effects:
            Processes uploaded files, extracts Aadhaar info, and updates session state.
        """
        aadhaar_info_list = []
        upload_type = st.selectbox("Select Upload Type", ["Image", "PDF"], key="upload_type_unique")
        if upload_type == "Image":
            aadhaar_files_img = st.file_uploader("Upload Aadhaar Images (Max 2)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="aadhaar_image_uploader")
            if len(aadhaar_files_img) == 2:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file1:
                        temp_file1.write(aadhaar_files_img[0].read())
                        temp_path1 = temp_file1.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file2:
                        temp_file2.write(aadhaar_files_img[1].read())
                        temp_path2 = temp_file2.name
                    output_pdf_path = os.path.join(tempfile.gettempdir(), "combined_aadhaar.pdf")
                    aadhaar_extractor = AadhaarExtractor()
                    aadhaar_extractor.combine_images_preserve_size(temp_path1, temp_path2, output_pdf_path)
                    with open(output_pdf_path, 'rb') as f:
                        pdf_bytes = f.read()
                        info = aadhaar_extractor.extract_from_pdf(pdf_bytes)
                        if info:
                            aadhaar_info_list.append(info)
                    os.unlink(temp_path1)
                    os.unlink(temp_path2)
                    if os.path.exists(output_pdf_path):
                        os.unlink(output_pdf_path)
                    st.success("Combined Aadhaar images processed.")
                except Exception as e:
                    st.warning(f"Error combining images: {e}")
            else:
                for img_file in aadhaar_files_img[:1]:
                    try:
                        img_file.seek(0)
                        image_bytes = img_file.read()
                        aadhaar_extractor = AadhaarExtractor()
                        info = aadhaar_extractor.extract_from_image(image_bytes)
                        if info:
                            aadhaar_info_list.append(info)
                    except Exception as e:
                        st.warning(f"Error processing image: {e}")
        elif upload_type == "PDF":
            aadhaar_file_pdf = st.file_uploader("Upload Aadhaar PDF", type=["pdf"], key="aadhaar_pdf_uploader")
            if aadhaar_file_pdf:
                try:
                    pdf_bytes = aadhaar_file_pdf.read()
                    aadhaar_extractor = AadhaarExtractor()
                    info = aadhaar_extractor.extract_from_pdf(pdf_bytes)
                    if info:
                        aadhaar_info_list.append(info)
                    else:
                        logger.warning("No Aadhaar info extracted from PDF")
                        st.warning("No Aadhaar info extracted from PDF")
                except Exception as e:
                    st.warning(f"Error processing PDF: {e}")
        if aadhaar_info_list:
            st.markdown("### Confirm and Edit Aadhaar Details")
            info = aadhaar_info_list[0]
            name = st.text_input("Name", value=info.get("name") or "", key="aadhaar_name_input")
            raw_dob = info.get("dob_or_yob")
            dob_value = date(2000, 1, 1)
            if raw_dob:
                try:
                    if len(raw_dob) == 4 and raw_dob.isdigit():
                        dob_value = date(int(raw_dob), 1, 1)
                    elif re.match(r'\d{2}[/-]\d{2}[/-]\d{4}', raw_dob):
                        dob_value = datetime.datetime.strptime(raw_dob, "%d-%m-%Y" if "-" in raw_dob else "%d/%m/%Y").date()
                    else:
                        st.warning(f"Invalid DOB format: {raw_dob}. Using default date.")
                except ValueError as e:
                    st.warning(f"Error parsing DOB '{raw_dob}': {e}. Using default date.")
            else:
                st.warning("No DOB found in extracted data. Using default date.")
            dob = st.date_input("Date of Birth", value=dob_value, key="aadhaar_dob_input")
            today = date.today()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            st.text_input("Age", value=str(age), disabled=True, key="aadhaar_age_input")
            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
                index=0 if str(info.get("gender", "")).lower() == "male" else 1,
                key="aadhaar_gender_select"
            )
            aadhar_no = st.text_input("Aadhaar Number", value=info.get("aadhar_no", "") or "", key="aadhaar_number_input")
            state_list = st.session_state.get("states_list", [])
            default_state = info.get("state") or (state_list[0] if state_list else "")
            state = st.selectbox(
                "State",
                options=state_list,
                index=state_list.index(default_state) if default_state in state_list else 0,
                key="aadhaar_state_select_unique"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit", key="aadhaar_submit_button"):
                    st.session_state.aadhaar_info_list = [{
                        "name": name,
                        "dob_or_yob": str(dob),
                        "age": age,
                        "gender": gender,
                        "aadhar_no": aadhar_no,
                        "state": state
                    }]
                    st.rerun()
            with col2:
                if st.button("Cancel", key="aadhaar_cancel_button"):
                    st.rerun()

    with st.container():
        col_upload, col_input, col_lang, col_voice = st.columns([0.6, 12, 2, 1])
        with col_upload:
            if st.button("➕", key="upload_button"):
                aadhaar_upload_dialog()
        with col_input:
            query = st.chat_input("Ask anything about government schemes...", key="chat_input")
        with col_lang:
            source_language = st.selectbox(
                "Language",
                options=[
                    ('en', 'English - English'),
                    ('hi', 'Hindi - हिंदी'),
                    ('bn', 'Bengali - বাংলা'), 
                    ('te', 'Telugu - తెలుగు'),
                    ('mr', 'Marathi - मराठी'),
                    ('ta', 'Tamil - தமிழ்'),
                    ('gu', 'Gujarati - ગુજરાતી'),
                    ('kn', 'Kannada - ಕನ್ನಡ'),
                    ('ml', 'Malayalam - മലയാളം'),
                    ('pa', 'Punjabi - ਪੰਜਾਬੀ'),
                    ('ur', 'Urdu - اردو'),
                    ('or', 'Odia - ଓଡ଼ିଆ'),
                    ('as', 'Assamese - অসমীয়া'),
                ],
                index=0,
                format_func=lambda x: x[1],
                key="source_language"
            )
        with col_voice:
            mic_audio = mic.mic_recorder(
                start_prompt="🎤 Record", 
                stop_prompt="⏹️ Stop", 
                key="voice_recorder",
                just_once=False,
                use_container_width=True,
                format="wav"
            )

    if query:
        if (st.session_state.last_llm_model != st.session_state.llm_model or 
            (st.session_state.last_source_language and 
             st.session_state.last_source_language != source_language[0])):
            st.info("Please enter a new query after changing the language or LLM model.")
            st.session_state.last_llm_model = st.session_state.llm_model
            st.session_state.last_source_language = source_language[0]
            return
        st.session_state.chat_history.append(("user", query))
        with chat_placeholder.container():
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)
                with st.chat_message("assistant", avatar="🧠"):
                    output_box = st.empty()
                    response_parts = []
                    english_query = query
                    if source_language[0] != 'en':
                        try:
                            english_query = translate_preserving_structure(
                                query,
                                source_language[0],
                                "en"
                            )
                            debug_log(f"English Query: {english_query}")
                        except Exception as e:
                            st.error(f"Translation failed: {e}")
                    selections = None
                    if dataset_type == 'State':
                        selections = st.session_state['state_selection_unique']
                    elif dataset_type == 'Central':
                        selections = st.session_state['central_selection_unique']
                    async def get_response():
                        history_size = st.session_state.conversation_memory_size * 2
                        history = st.session_state.chat_history[-history_size:]
                        async for chunk in st.session_state.retriever.generate_streaming(
                            query=english_query,
                            llm=st.session_state.llm,
                            filter_con=selections,
                            discovery_top_n=st.session_state.discovery_top_n,
                            use_multi_query=st.session_state.use_multi_query,
                            multi_query_n=st.session_state.multi_query_n if st.session_state.use_multi_query else None,
                            multi_query_ret_n=st.session_state.multi_query_ret_n if st.session_state.use_multi_query else None,
                            chat_history=history
                        ):
                            response_parts.append(chunk)
                            output_box.markdown("".join(response_parts))
                    await get_response()
                    full_response = "".join(response_parts)
                    debug_log(f"English Response: {full_response}")
                    if source_language[0] != 'en':
                        try:
                            with st.spinner():
                                full_response = translate_preserving_structure(
                                    full_response,
                                    "en",
                                    source_language[0]
                                )
                            output_box.markdown(full_response)
                        except Exception as e:
                            st.error(f"Error translating response: {e}")
                    st.session_state.chat_history.append(("assistant", full_response))
                    # if st.button("🔊 Dictate Response", key=f"dictate_text_{len(st.session_state.chat_history)}"):
                    #     with st.spinner("Converting response to speech..."):
                    #         try:
                    #             agent = BhashiniVoiceAgent(
                    #                 api_key=BHASHINI_API_KEY,
                    #                 user_id=BHASHINI_USER_ID,
                    #                 inference_api_key=BHASHINI_INFERENCE_API_KEY
                    #             )
                    #             tts_payload = {
                    #                 "pipelineTasks": [
                    #                     {
                    #                         "taskType": "tts",
                    #                         "config": {
                    #                             "language": {"sourceLanguage": source_language[0]},
                    #                             "gender": voice_gender,
                    #                             "samplingRate": 8000
                    #                         }
                    #                     }
                    #                 ],
                    #                 "inputData": {
                    #                     "input": [{"source": full_response}]
                    #                 }
                    #             }
                    #             tts_result = agent.call_pipeline(tts_payload)
                    #             audio_base64 = tts_result["pipelineResponse"][0]["audio"][0]["audioContent"]
                    #             st.audio(base64.b64decode(audio_base64), format="audio/wav")
                    #         except Exception as e:
                    #             st.error(f"Error converting to speech: {e}")

    if mic_audio:
        audio_hash = hashlib.md5(mic_audio['bytes']).hexdigest()
        if st.session_state.get('last_processed_audio_hash') != audio_hash:
            if (st.session_state.last_llm_model != st.session_state.llm_model or 
                (st.session_state.last_source_language and 
                st.session_state.last_source_language != source_language[0])):
                st.session_state.last_llm_model = st.session_state.llm_model
                st.session_state.last_source_language = source_language[0]
                return
            
            converted_audio = convert_audio_to_required_format(mic_audio['bytes'])
            if not converted_audio:
                st.error("Failed to convert audio. Please record a clear audio clip (at least 2 seconds) and try again. Ensure your microphone is working.")
                return
            if not validate_audio(converted_audio):
                st.error("Converted audio is invalid. Please record a clear audio clip (at least 2 seconds) in WAV format (16-bit, mono, 16kHz).")
                return
            
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(converted_audio)
                    tmp_path = tmp.name
                agent = BhashiniVoiceAgent(
                    api_key=BHASHINI_API_KEY,
                    user_id=BHASHINI_USER_ID,
                    inference_api_key=BHASHINI_INFERENCE_API_KEY
                )
                with st.spinner("Processing your voice input..."):
                    start_time = time.time()
                    result = agent.process_audio_pipeline(
                        audio_path=tmp_path,
                        source_lang=source_language[0]
                    )
                    response_time = time.time() - start_time
                    update_performance_stats(success=True, response_time=response_time, language=source_language[0])
                    if not result or not result.get('asr_text'):
                        st.error("Failed to transcribe audio. Please try again.")
                    elif result['asr_text'].strip():
                        st.session_state.chat_history.append(("user", result['asr_text']))
                        with chat_placeholder.container():
                            with chat_container:
                                with st.chat_message("user"):
                                    st.markdown(result['asr_text'])
                                with st.chat_message("assistant", avatar="🧠"):
                                    output_box = st.empty()
                                    response_parts = []
                                    selections = None
                                    if dataset_type == 'State':
                                        selections = st.session_state['state_selection_unique']
                                    elif dataset_type == 'Central':
                                        selections = st.session_state['central_selection_unique']
                                    with st.spinner("Getting response from LLM..."):
                                        async def get_voice_response():
                                            history_size = st.session_state.conversation_memory_size * 2
                                            history = st.session_state.chat_history[-history_size:]
                                            async for chunk in st.session_state.retriever.generate_streaming(
                                                query=result['english_query'],
                                                llm=st.session_state.llm,
                                                filter_con=selections,
                                                discovery_top_n=st.session_state.discovery_top_n,
                                                use_multi_query=st.session_state.use_multi_query,
                                                multi_query_n=st.session_state.multi_query_n if st.session_state.use_multi_query else None,
                                                multi_query_ret_n=st.session_state.multi_query_ret_n if st.session_state.use_multi_query else None,
                                                chat_history=history
                                            ):
                                                response_parts.append(chunk)
                                                output_box.markdown("".join(response_parts))
                                        await get_voice_response()
                                        full_response = "".join(response_parts)
                                        debug_log(f"English Query: {result['english_query']}")
                                        debug_log(f"English Response: {full_response}")
                                        try:
                                            if source_language[0] != 'en':
                                                full_response = translate_preserving_structure(
                                                    full_response,
                                                    "en",
                                                    source_language[0]
                                                )
                                            output_box.markdown(full_response)
                                            with st.spinner("Converting response to speech..."):
                                                tts_payload = {
                                                    "pipelineTasks": [
                                                        {
                                                            "taskType": "tts",
                                                            "config": {
                                                                "language": {"sourceLanguage": source_language[0]},
                                                                "gender": voice_gender,
                                                                "samplingRate": 8000
                                                            }
                                                        }
                                                    ],
                                                    "inputData": {
                                                        "input": [{"source": full_response}]
                                                    }
                                                }
                                                tts_result = agent.call_pipeline(tts_payload)
                                                audio_base64 = tts_result["pipelineResponse"][0]["audio"][0]["audioContent"]
                                                st.audio(base64.b64decode(audio_base64), format="audio/wav")
                                        except Exception as e:
                                            st.error(f"Error processing response: {e}")
                                        st.session_state.chat_history.append(("assistant", full_response))
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                update_performance_stats(success=False, response_time=0, language=source_language[0])
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.error(f"Error cleaning up temporary file: {e}")
                
        st.session_state.last_processed_audio_hash = audio_hash
    
    if st.session_state.last_source_language != source_language[0]:
        st.session_state.last_source_language = source_language[0]
    if st.session_state.last_llm_model != llm_model:
        st.session_state.last_llm_model = llm_model
        st.session_state.llm_model = llm_model
    if st.session_state.debug_mode:
        st.subheader("Debug Logs")
        for log in st.session_state.get('debug_logs', [])[-1:-6:-1]:
            st.code(log, language='text')

if __name__ == "__main__":
    asyncio.run(main())

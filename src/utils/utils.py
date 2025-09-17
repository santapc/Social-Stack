import os
import streamlit as st
import pickle
from src.config import LLM_MODELS, DATASET_NAMES

BHASHINI_API_KEY = os.getenv('BHASHINI_API_KEY')
BHASHINI_USER_ID = os.getenv('BHASHINI_USER_ID')
BHASHINI_INFERENCE_API_KEY = os.getenv("BHASHINI_INFERENCE_API_KEY")

def validate_bhashini_setup():
    """
    Validates the presence and format of Bhashini environment variables.
    Returns:
        bool: True if all environment variables are present and valid, False otherwise.
    Side Effects:
        Displays error or warning messages in the Streamlit app if validation fails.
    """
    required_vars = {
        'BHASHINI_API_KEY': BHASHINI_API_KEY,
        'BHASHINI_USER_ID': BHASHINI_USER_ID,
        'BHASHINI_INFERENCE_API_KEY': BHASHINI_INFERENCE_API_KEY
    }
    missing_vars = [key for key, value in required_vars.items() if not value]
    if missing_vars:
        return False
    else:
        return True

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
        # logger.info(f"Debug log: {msg}") # logger is not defined here

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

def initialize_session_state():
    """
    Initializes session state variables for the Streamlit application.
    Side Effects:
        Sets default values for session state variables if they don't exist, including:
        - debug_mode
        - debug_logs
        - bhashini_stats
        - chat_history
        - dataset settings
        - LLM settings
        - Aadhaar-related settings
    """
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'debug_logs' not in st.session_state:
        st.session_state.debug_logs = []
    if 'bhashini_stats' not in st.session_state:
        st.session_state.bhashini_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'language_usage': {}
        }
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_response_text' not in st.session_state:
        st.session_state.last_response_text = ""
    if 'dataset' not in st.session_state:
        st.session_state.dataset = DATASET_NAMES[0] if DATASET_NAMES else "default_dataset"
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = LLM_MODELS[0] if LLM_MODELS else "default_llm_model"
    if 'discovery_top_n' not in st.session_state:
        st.session_state.discovery_top_n = 5
    if 'use_multi_query' not in st.session_state:
        st.session_state.use_multi_query = False
    if 'multi_query_n' not in st.session_state:
        st.session_state.multi_query_n = 3
    if 'multi_query_ret_n' not in st.session_state:
        st.session_state.multi_query_ret_n = 3
    if 'states_list' not in st.session_state:
        st.session_state.states_list = pickle_read("src/assets/all_states.bin")
    if 'ministries_list' not in st.session_state:
        st.session_state.ministries_list = pickle_read("src/assets/all_ministries.bin")
    if 'last_llm_model' not in st.session_state:
        st.session_state.last_llm_model = st.session_state.llm_model
    if 'last_source_language' not in st.session_state:
        st.session_state.last_source_language = ""
    if 'aadhaar_info_list' not in st.session_state:
        st.session_state.aadhaar_info_list = []
    if 'aadhaar_state' not in st.session_state:
        st.session_state.aadhaar_state = None
    if 'aadhaar_filter_enabled' not in st.session_state:
        st.session_state.aadhaar_filter_enabled = False
    if 'sync_aadhaar' not in st.session_state:
        st.session_state.sync_aadhaar = False
    if 'dataset_type' not in st.session_state:
        st.session_state.dataset_type = "All"
    if 'user_info' not in st.session_state:
        st.session_state.user_info = None
    if "aadhaar_info_list" not in st.session_state:
        st.session_state.aadhaar_info_list = []
    if "aadhaar_state" not in st.session_state:
        st.session_state.aadhaar_state = None
    if "aadhaar_filter_enabled" not in st.session_state:
        st.session_state.aadhaar_filter_enabled = False
    if "sync_aadhaar" not in st.session_state:
        st.session_state.sync_aadhaar = False

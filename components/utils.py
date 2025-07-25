import os
import streamlit as st
import pickle

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
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.info("Please create a .env file with your Bhashini credentials")
        st.code("""
            # Create .env file in your project root:
            BHASHINI_API_KEY=your_api_key_here
            BHASHINI_USER_ID=your_user_id_here
            BHASHINI_INFERENCE_API_KEY=your_inference_key_here
        """)
        return False
    api_key = required_vars['BHASHINI_API_KEY']
    if len(api_key) < 30 or '-' not in api_key:
        st.warning("API key format seems incorrect. Please verify from Bhashini dashboard.")
    user_id = required_vars['BHASHINI_USER_ID']
    if len(user_id) < 30:
        st.warning("User ID format seems incorrect. Please verify from Bhashini dashboard.")
    return True

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
        st.session_state.dataset = "all_myschemes_simple_v1"
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = "llama-3.3-70b-versatile"
    if 'discovery_top_n' not in st.session_state:
        st.session_state.discovery_top_n = 5
    if 'use_multi_query' not in st.session_state:
        st.session_state.use_multi_query = False
    if 'multi_query_n' not in st.session_state:
        st.session_state.multi_query_n = 3
    if 'multi_query_ret_n' not in st.session_state:
        st.session_state.multi_query_ret_n = 3
    if 'states_list' not in st.session_state:
        st.session_state.states_list = pickle_read("assets/all_states.bin")
    if 'ministries_list' not in st.session_state:
        st.session_state.ministries_list = pickle_read("assets/all_ministries.bin")
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

def pickle_read(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

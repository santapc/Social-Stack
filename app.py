import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
import streamlit_mic_recorder as mic
import logging
import datetime
from datetime import date
import re
import tempfile
import base64
import hashlib
import time # Re-adding this import

from langchain_groq import ChatGroq # Re-adding this import




from src.components.advanced_retriever import SimpleRetriever
from src.services.llms import get_hpc_llm, get_hpc_llm_openai
from src.services.aadhaar_extractor import AadhaarExtractor
from src.config import LLM_MODELS, groq_models, gpt_models, ollama_models
from src.utils.utils import validate_bhashini_setup, initialize_session_state, debug_log, monitor_api_performance, update_performance_stats, pickle_read
from src.utils.helper_functions import get_embeddings
from src.services.bhashini import BhashiniVoiceAgent, show_bhashini_status as bhashini_show_status_service
from src.services.digilocker import DigilockerClient
from src.config import REDIRECT_URL
st.set_page_config(
    page_title="Social Stack",
    layout="wide",
    initial_sidebar_state="expanded"
)



load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def main():
    initialize_session_state()
    st.title("Social Stack: Transforming Welfare Delivery")
    query_params = st.query_params
    try:
        if os.environ["DIGILOCKER_CLIENT_ID"] and os.environ["DIGILOCKER_CLIENT_SECRET"] and os.environ["DIGILOCKER_CODE_VERIFIER"] and os.environ["DIGILOCKER_NSSO_URL"]:
            
            if 'user_info' not in st.session_state:
                st.session_state.user_info = None

            if 'code' in query_params:
                st.sidebar.success("You are logged in with DigiLocker!")
                st.sidebar.link_button("Logout", REDIRECT_URL)

                if not st.session_state.user_info:
                    digi = DigilockerClient(
                        client_id=os.environ["DIGILOCKER_CLIENT_ID"],
                        client_secret=os.environ["DIGILOCKER_CLIENT_SECRET"],
                        redirect_uri=REDIRECT_URL,
                        code_verifier=os.environ["DIGILOCKER_CODE_VERIFIER"]
                    )
                    
                    access_token = digi.get_access_token(query_params['code'])
                    user_info = digi.get_eaadhaar_info()
                    st.session_state.user_info = user_info
                else:
                    user_info = st.session_state.user_info
            else:
                user_info = st.session_state.user_info
                if st.sidebar.button("Login with DigiLocker"):
                    st.markdown(
                        f"""
                        <meta http-equiv="refresh" content="0; url={os.environ['DIGILOCKER_NSSO_URL']}">
                        """,
                        unsafe_allow_html=True,
                    )
    except KeyError:
        debug_log("DigiLocker environment variables not set. Skipping DigiLocker login.")
    





    
    bhashini_flag=validate_bhashini_setup()
    source_language=[]
    embeddings = get_embeddings()
    if bhashini_flag:
        try:
            bhashini_agent = BhashiniVoiceAgent(
                api_key=os.getenv('BHASHINI_API_KEY'),
                user_id=os.getenv('BHASHINI_USER_ID'),
                inference_api_key=os.getenv("BHASHINI_INFERENCE_API_KEY")
            )
        except KeyError:
            debug_log("Bhashini environment variables not set. Please configure them to use Bhashini services.")
    else:
        source_language.append("en")
        
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
            elif st.session_state.llm_model in ollama_models:
                st.session_state.llm = get_hpc_llm(model=st.session_state.llm_model)
            elif st.session_state.llm_model in gpt_models:
                st.session_state.llm = get_hpc_llm_openai(model=st.session_state.llm_model)
            st.session_state.active_llm_model = st.session_state.llm_model
        except Exception as e:
            if "AuthenticationError" in str(e):
                st.error("Authentication Error: Please check your OpenAI API key in the .env file.")
            else:
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

    st.sidebar.header("Configuration")
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


    try:
        if os.environ["DIGILOCKER_CLIENT_ID"] and os.environ["DIGILOCKER_CLIENT_SECRET"] and os.environ["DIGILOCKER_CODE_VERIFIER"] and os.environ["DIGILOCKER_NSSO_URL"]:
            with st.sidebar.expander("Profile", expanded=False):
                if st.session_state.user_info:
                    st.markdown("### User Info")
                    for k, v in st.session_state.user_info.items():
                        st.markdown(f"- **{k.capitalize()}**: {v}")

                    if st.sidebar.button("Sync with Aadhaar", key="sync_aadhaar_button_unique"):
                        aadhaar_state = st.session_state.user_info.get("state")
                        if aadhaar_state and aadhaar_state in st.session_state.states_list:
                            st.session_state.dataset_type = "State"
                            st.session_state['state_selection_unique'] = [aadhaar_state]
                            st.session_state['sync_aadhaar'] = True
                            st.session_state.aadhaar_state = aadhaar_state
                        else:
                            st.error("Aadhaar state not found in available states.")
                            st.session_state['sync_aadhaar'] = False
                else:
                    st.info("No user info available. Please log in with DigiLocker.")
    except KeyError:
        debug_log("DigiLocker environment variables not set. Skipping DigiLocker info display.")
        st.sidebar.info("DigiLocker environment variables not set.")

    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Settings")
    
    with st.sidebar.expander("Advanced LLM Settings"):
        llm_model = st.selectbox(
            "LLM Model",
            groq_models + gpt_models,
            index=LLM_MODELS.index(st.session_state.llm_model),
        )
        st.session_state.llm_model = llm_model
        st.session_state.discovery_top_n = st.slider("Default No. of Schemes", min_value=1, max_value=12, value=5)
        st.session_state.conversation_memory_size = st.slider("Conversation Memory Size", min_value=0, max_value=10, value=5)

    with st.sidebar.expander("Advanced Audio Settings"):
        voice_gender = st.selectbox("Voice Gender for TTS", ["female", "male"], index=0)

    st.session_state.debug_mode = st.sidebar.checkbox("Multilingual Debug Mode", value=st.session_state.debug_mode)


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
        if bhashini_flag:
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
                    start_prompt="🎤", 
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
                    if bhashini_flag:
                        if source_language[0] != 'en':
                            try:
                                english_query = bhashini_agent.translate_preserving_structure(
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
                        try:
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
                        except Exception as e:
                            error_message = str(e).lower()
                            if "api key" in error_message or "authentication" in error_message or "401" in error_message:
                                st.error("Error: LLM API key is incorrect or missing. Please check your API key configuration.")
                            else:
                                st.error(f"An unexpected error occurred: {e}")
                            return # Exit if an error occurs
                    await get_response()
                    full_response = "".join(response_parts)
                    if not full_response: # If an error occurred and no response was generated
                        return
                    debug_log(f"English Response: {full_response}")
                    if source_language[0] != 'en':
                        try:
                            with st.spinner():
                                full_response = bhashini_agent.translate_preserving_structure(
                                    full_response,
                                    "en",
                                    source_language[0]
                                )
                            output_box.markdown(full_response)
                        except Exception as e:
                            st.error(f"Error translating response: {e}")
                    st.session_state.chat_history.append(("assistant", full_response))
    if bhashini_flag:                
        if mic_audio:
            audio_hash = hashlib.md5(mic_audio['bytes']).hexdigest()
            if st.session_state.get('last_processed_audio_hash') != audio_hash:
                if (st.session_state.last_llm_model != st.session_state.llm_model or 
                    (st.session_state.last_source_language and 
                    st.session_state.last_source_language != source_language[0])):
                    st.session_state.last_llm_model = st.session_state.llm_model
                    st.session_state.last_source_language = source_language[0]
                    return
                
                try:
                    with st.spinner("Processing your voice input..."):
                        start_time = time.time()
                        result = bhashini_agent.process_audio_pipeline(
                            audio_bytes=mic_audio['bytes'],
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
                                                try:
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
                                                except Exception as e:
                                                    error_message = str(e).lower()
                                                    if "api key" in error_message or "authentication" in error_message or "401" in error_message:
                                                        st.error("Error: LLM API key is incorrect or missing. Please check your API key configuration.")
                                                    else:
                                                        st.error(f"An unexpected error occurred: {e}")
                                                    return # Exit if an error occurs
                                            await get_voice_response()
                                            full_response = "".join(response_parts)
                                            if not full_response: # If an error occurred and no response was generated
                                                return
                                            debug_log(f"English Query: {result['english_query']}")
                                            debug_log(f"English Response: {full_response}")
                                            try:
                                                if source_language[0] != 'en':
                                                    full_response = bhashini_agent.translate_preserving_structure(
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
                                                    tts_result = bhashini_agent.call_pipeline(tts_payload)
                                                    audio_base64 = tts_result["pipelineResponse"][0]["audio"][0]["audioContent"]
                                                    st.audio(base64.b64decode(audio_base64), format="audio/wav")
                                            except Exception as e:
                                                st.error(f"Error processing response: {e}")
                                            st.session_state.chat_history.append(("assistant", full_response))
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    update_performance_stats(success=False, response_time=0, language=source_language[0])
                    
            st.session_state.last_processed_audio_hash = audio_hash
    # if bhashini_flag:
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

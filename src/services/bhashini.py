import requests
import base64
import logging
import time
from urllib3.exceptions import MaxRetryError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import certifi
from pydub import AudioSegment
from pydub.utils import which
import io
import tempfile # Ensure tempfile is imported

AudioSegment.converter = which("ffmpeg")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

callback_url = os.getenv('callback_url')
pipeline_id = os.getenv('BHASHINI_PIPELINE_ID')

class BhashiniVoiceAgent:
    def __init__(self, api_key: str, user_id: str, inference_api_key: str):
        self.api_key = api_key
        self.user_id = user_id
        self.inference_api_key = inference_api_key
        self.callback_url = callback_url
        self.pipeline_id = pipeline_id

        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(
            total=10,  # Increased retries
            backoff_factor=3,  # Longer backoff
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            respect_retry_after_header=True
        ))
        self.session.mount("https://", adapter)
        self.timeout = (60, 120)  # Increased timeouts: 60s connect, 120s read

    def call_pipeline(self, payload):
        """
        Calls the Bhashini pipeline API with the given payload.
        """
        headers = {
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'User-ID': self.user_id,
            'API-Key': self.api_key,
            'Authorization': self.inference_api_key
        }
        try:
            response = self.session.post(
                self.callback_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                verify=certifi.where()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Bhashini API call failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Server response: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Bhashini API call failed: {e}")

    def validate_audio(self, audio_path):
        """Validate audio file format."""
        try:
            audio = AudioSegment.from_file(audio_path, format="wav")
            if audio.frame_rate not in [8000, 16000]:
                raise ValueError(f"Unsupported sample rate: {audio.frame_rate}. Expected 8kHz or 16kHz.")
            if audio.sample_width != 2:
                raise ValueError(f"Unsupported sample width: {audio.sample_width}. Expected 16-bit.")
            if audio.channels != 1:
                raise ValueError(f"Unsupported channels: {audio.channels}. Expected mono.")
            logger.info(f"Audio validation: Sample rate={audio.frame_rate}, Channels={audio.channels}, Sample width={audio.sample_width}, Duration={len(audio)/1000}s")
            return True
        except Exception as e:
            logger.error(f"Invalid audio file: {str(e)}")
            return False

    def encode_audio(self, audio_path):
        """Encode audio file to base64."""
        if not self.validate_audio(audio_path):
            raise ValueError("Audio file validation failed.")
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        encoded = base64.b64encode(audio_data).decode('utf-8')
        logger.info(f"Encoded audio size: {len(encoded)} bytes")
        return encoded

    def test_bhashini_connection(self):
        """
        Tests the connection to the Bhashini API using a silent audio sample.

        Returns:
            dict: Status and message indicating the health of the Bhashini connection.

        Raises:
            Exception: If the API call fails, returns an error message in the status dict.
        """
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                audio = AudioSegment.silent(duration=1000, frame_rate=16000)
                audio = audio.set_channels(1).set_sample_width(2)
                audio.export(tmp.name, format="wav")
                audio_b64 = self.encode_audio(tmp.name)
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
            response = self.call_pipeline(test_payload)
            os.unlink(tmp.name)
            return {"status": "healthy", "message": "Bhashini connection successful"}
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Connection test failed: {error_msg}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Server response: {e.response.status_code} - {e.response.text}")
            return {"status": "unhealthy", "message": error_msg}

    def detect_language(self, text):
        """
        Detects the language of the input text using Bhashini API.

        Args:
            text (str): Text to analyze for language detection.

        Returns:
            str: The detected language code (e.g., 'en', 'hi').

        Raises:
            Exception: If language detection fails.
        """
        payload = {
            "pipelineTasks": [
                {
                    "taskType": "language-detection",
                    "config": {
                        "language": {
                            "sourceLanguage": "auto"
                        }
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text}]
            }
        }

        try:
            res = self.call_pipeline(payload)
            lang = res["pipelineResponse"][0]["output"][0]["source"]
            logger.info(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise Exception(f"Language detection failed: {e}")

    def translate_text(self, text, source_lang, target_lang):
        """
        Translates text using Bhashini API.

        Args:
            text (str): Text to translate.
            source_lang (str): Source language code (e.g., 'hi' for Hindi).
            target_lang (str): Target language code (e.g., 'en' for English).

        Returns:
            str: Translated text.

        Raises:
            Exception: If translation fails.
        """
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
                "input": [{"source": text}]
            }
        }
        translated = self.call_pipeline(translation_payload)
        result = translated["pipelineResponse"][0]["output"][0]["target"]
        return result

    def preserve_urls_during_translation(self, text, source_lang, target_lang):
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
            translated_text = self.translate_text(text_with_placeholders, source_lang, target_lang)
            for i, (url_type, link_text, url) in enumerate(urls):
                placeholder = f"URL_PLACEHOLDER_{i}"
                if url_type == 'markdown':
                    translated_text = translated_text.replace(placeholder, f"[{link_text}]({url})")
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

    def format_markdown_links(self, text):
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

    def translate_preserving_structure(self, text, source_lang, target_lang):
        import re # Import re here to avoid circular dependency if moved to top
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
                    translated = self.call_pipeline(translation_payload)
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

                translated_name = self.translate_text(link_text, source_lang, target_lang)
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

    def validate_audio_bytes(self, audio_bytes):
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

    def convert_audio_to_required_format(self, audio_bytes):
        """
        Converts audio to the required format for Bhashini (16-bit, mono, 16kHz WAV).

        Args:
            audio_bytes (bytes): Raw audio data.

        Returns:
            bytes: Converted audio bytes, or None if conversion fails.

        Raises:
            ValueError: If the audio data is invalid or too short.
        """
        debug_path = None # Initialize debug_path
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
            if debug_path: # Only log debug_path if it was successfully assigned
                logger.error(f"Debug audio saved at: {debug_path}")
            return None

    def process_audio_pipeline(self, audio_bytes, source_lang="hi"):
        """Main pipeline: ASR -> translate to EN -> LLM -> translate back -> TTS."""
        logger.info("Starting full voice pipeline...")
        tmp_path = None
        debug_path = None # Initialize debug_path here
        try:
            converted_audio = self.convert_audio_to_required_format(audio_bytes)
            if not converted_audio:
                raise Exception("Failed to convert audio to required format.")
            if not self.validate_audio_bytes(converted_audio):
                raise Exception("Converted audio is invalid.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(converted_audio)
                tmp_path = tmp.name
            
            audio_b64 = self.encode_audio(tmp_path)

            # Step 1: ASR
            logger.info("Running ASR...")
            asr_payload = {
                "pipelineTasks": [
                    {
                        "taskType": "asr",
                        "config": {
                            "language": {
                                "sourceLanguage": source_lang
                            }
                        }
                    }
                ],
                "inputData": {
                    "audio": [{"audioContent": audio_b64}]
                }
            }
            result = self.call_pipeline(asr_payload)

            asr_text = None
            for task in result.get("pipelineResponse", []):
                if task["taskType"] == "asr":
                    asr_text = task.get("output", [{}])[0].get("source", "")
            if not asr_text:
                raise Exception("ASR failed: No transcription returned")

            logger.info(f"ASR Result: {asr_text}")

            # Step 2: Translate to English
            logger.info("Translating to English...")
            english_query = self.translate_text(asr_text, source_lang, "en")
            logger.info(f"Translated to English: {english_query}")

            return {
                "asr_text": asr_text,
                "english_query": english_query,
            }
        except Exception as e:
            logger.error(f"Error in process_audio_pipeline: {str(e)}")
            raise Exception(f"Pipeline processing failed: {str(e)}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logger.error(f"Error cleaning up temporary file: {e}")

def show_bhashini_status(bhashini_agent):
    """
    Displays the status of the Bhashini API connection in the Streamlit UI.

    Args:
        bhashini_agent (BhashiniVoiceAgent): An instance of the BhashiniVoiceAgent.

    Returns:
        bool: True if the connection is healthy or in fallback mode, False otherwise.

    Side Effects:
        Displays status messages in the Streamlit app.
    """
    import streamlit as st # Import streamlit here to avoid circular dependency if moved to top
    if 'bhashini_status' not in st.session_state:
        with st.spinner("Testing Bhashini connection..."):
            st.session_state.bhashini_status = bhashini_agent.test_bhashini_connection()
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

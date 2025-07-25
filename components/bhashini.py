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

    def detect_language(self, text):
        """Detect the language of the input text."""
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

        headers = {
            "Authorization": self.inference_api_key,
            "Content-Type": "application/json"
        }

        try:
            res = self.session.post(self.callback_url, headers=headers, json=payload, timeout=self.timeout)
            res.raise_for_status()
            try:
                lang = res.json()["pipelineResponse"][0]["output"][0]["source"]
                logger.info(f"Detected language: {lang}")
                return lang
            except (KeyError, IndexError) as e:
                logger.error(f"Failed to parse language detection response: {e}")
                logger.error(f"Response: {res.text}")
                raise Exception("Failed to parse language detection response")
        except requests.exceptions.RequestException as e:
            logger.error(f"Language detection failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Server response: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Language detection failed: {str(e)}")

    def call_pipeline(self, payload):
        """Call the Bhashini API with the given payload."""
        headers = {
            "Authorization": self.inference_api_key,
            "Content-Type": "application/json",
            "User-Agent": "BhashiniVoiceAgent/1.0"
        }

        try:
            logger.info(f"Sending payload to {self.callback_url}")
            res = self.session.post(
                self.callback_url, headers=headers, json=payload, timeout=self.timeout, verify=certifi.where()
            )
            res.raise_for_status()
            logger.info(f"API response: {res.status_code} ")
            return res.json()
        except MaxRetryError as e:
            logger.error(f"MaxRetryError: Too many 500 responses. Last response: {getattr(e, 'response', 'No response')}")
            raise Exception("Bhashini API failed repeatedly with HTTP 500.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Pipeline request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Server response: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Pipeline processing failed: {str(e)}")

    def process_audio_pipeline(self, audio_path, source_lang="hi", target_lang="en"):
        """Main pipeline: ASR -> translate to EN -> LLM -> translate back -> TTS."""
        logger.info("Starting full voice pipeline...")
        try:
            time.sleep(2)  # Add delay to avoid rate limiting
            audio_b64 = self.encode_audio(audio_path)

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
                    "input": [{"source": asr_text}]
                }
            }
            translated = self.call_pipeline(translation_payload)
            english_query = translated["pipelineResponse"][0]["output"][0]["target"]
            logger.info(f"Translated to English: {english_query}")

            # Step 3: LLM (mock)
            logger.info("Calling LLM...")
            llm_response = f"This is a response to: {english_query}"
            logger.info(f"LLM Response: {llm_response}")

            # Step 4: Back Translate to original language
            logger.info("Back-translating to original language...")
            back_payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "language": {
                                "sourceLanguage": "en",
                                "targetLanguage": source_lang
                            }
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": llm_response}]
                }
            }
            back_translated = self.call_pipeline(back_payload)
            final_text = back_translated["pipelineResponse"][0]["output"][0]["target"]
            logger.info(f"Final translated text: {final_text}")

            # Step 5: TTS
            logger.info("Converting to speech...")
            tts_payload = {
                "pipelineTasks": [
                    {
                        "taskType": "tts",
                        "config": {
                            "language": {"sourceLanguage": source_lang},
                            "gender": "female",
                            "samplingRate": 8000
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": final_text}]
                }
            }
            tts_result = self.call_pipeline(tts_payload)
            audio_base64 = tts_result["pipelineResponse"][0]["audio"][0]["audioContent"]
            logger.info("TTS completed.")

            return {
                "asr_text": asr_text,
                "english_query": english_query,
                "llm_response": llm_response,
                "final_response_text": final_text,
                "tts_audio_base64": audio_base64
            }
        except Exception as e:
            logger.error(f"Error in process_audio_pipeline: {str(e)}")
            raise Exception(f"Pipeline processing failed: {str(e)}")

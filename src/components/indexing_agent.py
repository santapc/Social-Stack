import os
import json
import re
import logging
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple
import pytesseract
from pdf2image import convert_from_path
from src.services.llms import get_hpc_llm
from src.components.prompts import get_scheme_extraction_prompt_template

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def pickle_read(filename: str) -> List[str]:
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error reading pickle file {filename}: {str(e)}")
        return []

valid_states = pickle_read("assets/all_states.bin")
valid_ministries = pickle_read("assets/all_ministries.bin")

class Agent:
    def __init__(self, model, chunk_size: int = None):
        self.llm = model
        # Combine valid states and ministries, store as lowercase for case-insensitive comparison
        self.valid_categories = [cat.lower() for cat in (valid_states + valid_ministries)]
        self.chunk_size = chunk_size  # None means dynamic chunk size

    def calculate_chunk_size(self, total_pages: int) -> int:
        """Calculate dynamic chunk size based on total pages."""
        if total_pages <= 25:
            return max(1, total_pages // 2)
        elif total_pages <= 50:
            return max(3, total_pages // 4)
        else:
            return max(5, total_pages // 8)

    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        try:
            reader = PdfReader(file_path)
            pages_text = []
            ocr_images = None
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                if not text.strip():
                    logger.warning(f"No extractable text on page {page_num}, attempting OCR...")
                    if ocr_images is None:
                        try:
                            ocr_images = convert_from_path(file_path)
                        except Exception as e:
                            logger.error(f"Failed to convert PDF to images for OCR: {str(e)}")
                            pages_text.append((page_num, ""))
                            continue
                    try:
                        image = ocr_images[page_num - 1]
                        ocr_text = pytesseract.image_to_string(image, lang='eng')
                        logger.info(f"OCR extracted {len(ocr_text)} characters from page {page_num}")
                        pages_text.append((page_num, ocr_text))
                    except Exception as e:
                        logger.error(f"OCR failed for page {page_num}: {str(e)}")
                        pages_text.append((page_num, ""))
                else:
                    logger.info(f"Extracted text from page {page_num}: {len(text)} characters")
                    pages_text.append((page_num, text))
            if not any(text for _, text in pages_text):
                raise ValueError("No extractable text found in any page of the PDF")
            return pages_text
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def process_chunk(self, chunk_pages: List[Tuple[int, str]]) -> Dict:
        if not any(text.strip() for _, text in chunk_pages):
            logger.warning(f"Skipping empty chunk: {chunk_pages}")
            return {}

        chunk_text = "\n".join(text for _, text in chunk_pages if text.strip())
        if not chunk_text:
            return {}

        prompt_str = get_scheme_extraction_prompt_template().format(
            text=chunk_text,
            valid_categories=", ".join(self.valid_categories)
        )
        logging.debug(f"Scheme extraction prompt: {prompt_str}")
        prompt = PromptTemplate.from_template(prompt_str)

        chain = prompt | self.llm
        result = chain.invoke({"text": chunk_text, "valid_categories": ", ".join(self.valid_categories)})
        response_str = result.content if hasattr(result, "content") else str(result)

        logger.info(f"Raw LLM response for chunk {chunk_pages[0][0]}-{chunk_pages[-1][0]}: {response_str}")

        try:
            cleaned_response = re.sub(r'```(?:json)?\n?', '', response_str, flags=re.MULTILINE)
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_data = json.loads(json_str)
                required_keys = [
                    "scheme_id", "scheme_name", "scheme_type", "scheme_category", "source_url",
                    "details", "benefits", "eligibility", "exclusions", "application_process", "documents_required"
                ]
                if not all(key in json_data for key in required_keys):
                    logger.error(f"Missing required keys in JSON for chunk {chunk_pages[0][0]}-{chunk_pages[-1][0]}: {json_data}")
                    return {}
                json_data["scheme_category"]=json_data["scheme_category"].replace("&","and")
                # Case-insensitive validation for scheme_category
                if json_data["scheme_category"].lower() not in self.valid_categories:
                    logger.error(f"""nvalid scheme_category in chunk {chunk_pages[0][0]}-{chunk_pages[-1][0]}: {json_data["scheme_category"].lower().replace("&","and") }""")
                    json_data["scheme_category"] = ""
                
                # Clean up fields to remove unwanted phrases and ensure string type
                for key in required_keys:
                    value = json_data[key]
                    if not isinstance(value, str):
                        json_data[key] = json.dumps(value) if value else ""
                    # Remove phrases like "Not provided", "None information", etc.
                    json_data[key] = re.sub(
                        r'(?i)\b(not\s+(?:provided|explicitly\s+mentioned|available|specified)|n/a|unknown|none\s+information|no\s+information|no\s+relevant\s+info)\b',
                        '',
                        json_data[key]
                    ).strip()

                return json_data
            else:
                logger.error(f"No JSON object found in response for chunk {chunk_pages[0][0]}-{chunk_pages[-1][0]}: {cleaned_response}")
                return {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON for chunk {chunk_pages[0][0]}-{chunk_pages[-1][0]}: {cleaned_response}")
            return {}

    def combine_results(self, chunk_results: List[Dict]) -> Dict:
        if not chunk_results:
            logger.error("No valid chunk results to combine")
            return {
                "scheme_id": "",
                "scheme_name": "",
                "scheme_type": "",
                "scheme_category": "",
                "source_url": "",
                "details": "",
                "benefits": "",
                "eligibility": "",
                "exclusions": "",
                "application_process": "",
                "documents_required": ""
            }

        combined = {
            "scheme_id": "",
            "scheme_name": "",
            "scheme_type": "",
            "scheme_category": "",
            "source_url": "",
            "details": "",
            "benefits": "",
            "eligibility": "",
            "exclusions": "",
            "application_process": "",
            "documents_required": ""
        }

        # Count frequency of scheme_category values
        category_counts = {}
        for result in chunk_results:
            if not result:
                continue
            
            category = str(result.get("scheme_category", "").replace("&","and")) if result.get("scheme_category").replace("&","and") is not None else ""
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1

        # Select the most frequent scheme_category (if any)
        if category_counts:
            most_frequent_category = max(category_counts, key=category_counts.get)
            combined["scheme_category"] = most_frequent_category

        # Process other fields
        for result in chunk_results:
            if not result:
                continue
            for key in combined:
                if key == "scheme_category":  # Skip scheme_category as it's already set
                    continue
                result_value = str(result.get(key, "")) if result.get(key) is not None else ""
                if result_value and not combined[key]:
                    combined[key] = result_value
                elif result_value and combined[key] and key in [
                    "details", "benefits", "eligibility", "exclusions", "application_process", "documents_required"
                ]:
                    combined[key] = f"{combined[key]}\n{result_value}".strip()

        if not combined["scheme_id"] and not combined["scheme_name"]:
            logger.error("Combined result lacks scheme_id and scheme_name")
            combined["details"] = "Incomplete scheme data extracted from PDF"

        logger.info(f"Combined result: {combined}")
        return combined
    def pdf_tool_func(self, file_path: str) -> Dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        if not file_path.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF")

        pages_text = self.extract_text_from_pdf(file_path)
        chunk_size = self.chunk_size if self.chunk_size is not None else self.calculate_chunk_size(len(pages_text))
        logger.info(f"Using chunk size: {chunk_size} for {len(pages_text)} pages")

        chunk_results = []
        for i in range(0, len(pages_text), chunk_size):
            chunk = pages_text[i:i + chunk_size]
            result = self.process_chunk(chunk)
            if result:
                chunk_results.append(result)

        combined_result = self.combine_results(chunk_results)
        return combined_result

    def run(self, input_text: str) -> Dict:
        logger.info(f"Agent.run input: {input_text}")
        try:
            file_path_match = re.search(r'Extract scheme details from (.+)', input_text)
            if not file_path_match:
                raise ValueError("Invalid input format. Expected: 'Extract scheme details from <file_path>'")
            file_path = file_path_match.group(1)
            
            result = self.pdf_tool_func(file_path)
            logger.info(f"Tool result: {result}")
            return {"output": result}
        except Exception as e:
            logger.error(f"Error executing agent: {str(e)}")
            raise
import pytesseract
import re
import string
import io
from PIL import Image
from difflib import SequenceMatcher
from pdf2image import convert_from_bytes
from datetime import date, datetime
from src.config import TESSERACT_PATH, POPPLER_PATH

# Set up paths (adjust these based on your environment)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# List of Indian states and UTs
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands",
    "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir",
    "Ladakh", "Lakshadweep", "Puducherry"
]

class AadhaarExtractor:
    def __init__(self):
        self.states = INDIAN_STATES

    def clean_line(self, line):
        """Clean a line of text by converting to lowercase and removing punctuation."""
        return line.lower().strip().translate(str.maketrans('', '', string.punctuation))

    def find_best_state_match(self, text, threshold=0.6):
        """Find the best matching Indian state using fuzzy matching."""
        best_state = None
        best_ratio = 0
        text = text.lower()
        for state in self.states:
            ratio = SequenceMatcher(None, text, state.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_state = state
        if best_ratio >= threshold:
            return best_state
        return None

    def calculateAge(self, dob_or_yob):
        """Calculate age based on DOB or YOB."""
        today = date.today()  # Current date: May 30, 2025
        try:
            if dob_or_yob and re.match(r'\d{2}[-/]\d{2}[-/]\d{4}', dob_or_yob):
                # Full date: DD-MM-YYYY or DD/MM/YYYY
                dob = datetime.strptime(dob_or_yob, "%d-%m-%Y" if '-' in dob_or_yob else "%d/%m/%Y").date()
                age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                return age if 0 <= age <= 120 else None  # Validate reasonable age
            elif dob_or_yob and re.match(r'\d{4}', dob_or_yob):
                # Year only: YYYY
                yob = int(dob_or_yob)
                age = 2025 - yob
                return age if 0 <= age <= 120 else None  # Validate reasonable age
        except ValueError:
            pass
        return None

    def extract_details(self, text: str) -> dict:
        """Extract Aadhaar details from text, including age."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        extracted = {
            "name": None,
            "dob_or_yob": None,
            "age": None,
            "gender": None,
            "aadhar_no": None,
            "state": None
        }

        # Aadhaar Number
        aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)
        if aadhaar_match:
            extracted["aadhar_no"] = aadhaar_match.group()

        # DOB or YOB
        dob_match = re.search(r'(?:DOB|Date of Birth)[^\d]*(\d{2}[\/\-.]\d{2}[\/\-.]\d{4})', text, re.IGNORECASE)
        yob_match = re.search(r'(?:YOB|Year of Birth)?[^\d]*(\d{4})', text, re.IGNORECASE)

        if dob_match:
            extracted["dob_or_yob"] = dob_match.group(1)
        elif yob_match:
            extracted["dob_or_yob"] = yob_match.group(1)
        else:
            year_only = re.search(r'\b(19|20)\d{2}\b', text)
            if year_only:
                extracted["dob_or_yob"] = year_only.group()

        # Calculate Age
        if extracted["dob_or_yob"]:
            extracted["age"] = self.calculateAge(extracted["dob_or_yob"])
        else:
            full_date_match = re.search(r'\b\d{2}[\/\-.]\d{2}[\/\-.]\d{4}\b', text)
            year_only_match = re.search(r'\b(19|20)\d{2}\b', text)
            if full_date_match:
                extracted["dob_or_yob"] = full_date_match.group()
            elif year_only_match:
                extracted["dob_or_yob"] = year_only_match.group()


        # Gender
        gender_patterns = {
            "Male": r'\b(MALE|पुरुष)\b',
            "Female": r'\b(FEMALE|महिला)\b',
            "Other": r'\b(OTHER|अन्य)\b'
        }
        for gender, pattern in gender_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                extracted["gender"] = gender.title()
                break

        # Name
        dob_index = -1
        for i, line in enumerate(lines):
            if extracted["dob_or_yob"] and extracted["dob_or_yob"] in line:
                dob_index = i
                break
            if re.search(r'(dob|yob|birth)', line, re.IGNORECASE):
                dob_index = i
                break

        skip_keywords = ['government', 'india', 'dob', 'yob', 'male', 'female', 'birth', 'year', 'unique', 'identification']
        name_candidates = []

        for line in lines:
            if any(char.isdigit() for char in line):
                continue
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue
            if 2 <= len(line.split()) <= 4:
                name_candidates.append(line)

        if dob_index > 0:
            for i in range(dob_index - 1, -1, -1):
                candidate = lines[i].strip()
                if not candidate or any(char.isdigit() for char in candidate):
                    continue
                if any(keyword in candidate.lower() for keyword in skip_keywords):
                    continue
                if 2 <= len(candidate.split()) <= 4:
                    extracted["name"] = candidate
                    break

        if not extracted["name"] and name_candidates:
            extracted["name"] = name_candidates[0]

        # State (fuzzy)
        matched_state = self.find_best_state_match(text)
        if not matched_state:
            for line in lines:
                cleaned = self.clean_line(line)
                matched_state = self.find_best_state_match(cleaned)
                if matched_state:
                    break

        extracted["state"] = matched_state

        return extracted

    def extract_from_pdf(self, pdf_bytes: bytes) -> dict:
        """Extract Aadhaar details from a PDF file."""
        try:
            images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_PATH)
            all_text = ""
            for img in images:
                all_text += pytesseract.image_to_string(img) + "\n"
            return self.extract_details(all_text)
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            return {}

    def extract_from_image(self, image_bytes: bytes) -> dict:
        """Extract Aadhaar details from an image."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
            return self.extract_details(text)
        except Exception as e:
            print(f"Error extracting from image: {e}")
            return {}

    def combine_images_preserve_size(self, img1_path: str, img2_path: str, output_pdf_path: str):
        """Combine two images into a single PDF without resizing."""
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            combined_width = img1.width + img2.width
            max_combined_width = max(img1.width, img2.width) * 2

            if combined_width <= max_combined_width:
                combined_height = max(img1.height, img2.height)
                new_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
                new_img.paste(img1, (0, 0))
                new_img.paste(img2, (img1.width, 0))
                new_img.save(output_pdf_path)
            else:
                img1.save(output_pdf_path, save_all=True, append_images=[img2])
        except Exception as e:
            print(f"Error combining images: {e}")
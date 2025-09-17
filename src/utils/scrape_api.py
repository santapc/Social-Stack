import requests
import os
import json
import logging
import time
from requests.exceptions import RequestException, HTTPError
from dotenv import load_dotenv
import glob
import re

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheme_fetching_cleaning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variables with relative paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'assets\scheme_data'))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
SEARCH_DIR = os.path.join(BASE_DIR, "search")
ERROR_DIR = os.path.join(BASE_DIR, "errors")
TARGET_SCHEME_FILE = os.path.join(CACHE_DIR, "target_scheme_dt.json")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(ERROR_DIR, exist_ok=True)

# API configuration
API_HEADERS = {
    "x-api-key": os.getenv("MYSCHEMES_BROWSER_API_KEY"),
    "accept": "application/json, text/plain, */*",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://www.myscheme.gov.in",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1"
}

def clean_text(text):
    """Clean text by replacing Unicode escape sequences and normalizing whitespace."""
    if not text or not isinstance(text, str):
        return text
    text = text.replace('\u20b9', '₹').replace('\u201c', '"').replace('\u201d', '"').replace('\u2019', "'")
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def extract_text_from_node(node):
    """Recursively extract readable text from any node (table, paragraph, list, etc.)."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        node_type = node.get('type')
        if node_type == 'table':
            rows = []
            for row in node.get('children', []):
                if row.get('type') == 'table_row':
                    cells = [extract_text_from_node(cell) for cell in row.get('children', []) if cell.get('type') == 'table_cell']
                    rows.append(' | '.join(cells))
            return '\n'.join(rows)
        elif node_type in ['paragraph', 'align_justify']:
            # Join all children recursively
            return ' '.join([extract_text_from_node(child) for child in node.get('children', [])])
        elif node_type == 'list_item':
            return ' '.join([extract_text_from_node(child) for child in node.get('children', [])])
        elif node_type == 'table_cell':
            return ' '.join([extract_text_from_node(child) for child in node.get('children', [])])
        elif 'text' in node:
            return clean_text(node['text'])
        else:
            # Fallback: join all children
            return ' '.join([extract_text_from_node(child) for child in node.get('children', [])]) if 'children' in node else ''
    if isinstance(node, list):
        return '\n'.join([extract_text_from_node(item) for item in node])
    return ''


def fetch_documents_required(scheme_id):
    """Fetch and extract documents required for a scheme from the documents API endpoint."""
    import requests
    API_URL = f"https://api.myscheme.gov.in/schemes/v5/public/schemes/{scheme_id}/documents"
    params = {"lang": "en"}
    try:
        response = requests.get(API_URL, headers=API_HEADERS, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Handle both possible structures
        documents = data.get('data', [])
        doc_lines = []
        if isinstance(documents, dict):
            # New structure: {'_id': ..., 'en': {'documents_required': [...], ...}}
            en = documents.get('en', {})
            doc_list = en.get('documents_required', [])
            for doc in doc_list:
                text = extract_text_from_node(doc)
                if text:
                    doc_lines.append(text)
            # Fallback to documentsRequired_md if present
            if not doc_lines:
                md = en.get('documentsRequired_md', '')
                if md:
                    doc_lines.append(clean_text(md))
        elif isinstance(documents, list):
            # Old structure: list of dicts with documentName, description, etc.
            for doc in documents:
                name = doc.get('documentName', '')
                desc = doc.get('description', '')
                mandatory = doc.get('mandatory', False)
                line = f"{'*' if mandatory else '-'} {name}: {desc}" if desc else f"{'*' if mandatory else '-'} {name}"
                doc_lines.append(line)
        return '\n'.join(doc_lines)
    except Exception as e:
        logger.error(f"Error fetching documents required for scheme {scheme_id}: {e}")
        return ''


def clean_scheme_data(raw_data, slug):
    """Clean and simplify scheme data from v5 API response. Ensures 'slug' is present in the output."""
    cleaned_data = {}
    try:
        if not isinstance(raw_data, dict):
            logger.error(f"Invalid scheme data for {slug}: raw_data is not a dictionary")
            with open(os.path.join(ERROR_DIR, f"error_raw_data_{slug}.json"), 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2)
            return None

        if 'data' not in raw_data or raw_data['data'] is None:
            logger.error(f"Invalid scheme data for {slug}: 'data' key missing or None")
            with open(os.path.join(ERROR_DIR, f"error_raw_data_{slug}.json"), 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2)
            return None

        data = raw_data['data'].get('en') if isinstance(raw_data['data'], dict) else None
        if data is None:
            logger.error(f"Invalid scheme data for {slug}: 'en' key missing or None")
            with open(os.path.join(ERROR_DIR, f"error_raw_data_{slug}.json"), 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, indent=2)
            return None

        basic_details = data.get('basicDetails', {}) if isinstance(data.get('basicDetails'), dict) else {}
        scheme_content = data.get('schemeContent', {}) if isinstance(data.get('schemeContent'), dict) else {}
        eligibility = data.get('eligibilityCriteria', {}) if isinstance(data.get('eligibilityCriteria'), dict) else {}
        application_process = data.get('applicationProcess', []) if isinstance(data.get('applicationProcess'), list) else []

        # Defensive: always use .get and check types, fallback to default
        # --- Ensure slug is present ---
        extracted_slug = raw_data.get('slug', '') or raw_data.get('data', {}).get('slug', '') or slug

        cleaned_data = {
            'scheme_id': raw_data.get('data', {}).get('_id', ''),
            'slug': extracted_slug,
            'scheme_name': clean_text(basic_details.get('schemeName', '')),
            'scheme_short_title': clean_text(basic_details.get('schemeShortTitle', '')),
            'state': (basic_details.get('state') or {}).get('label', '') if isinstance(basic_details.get('state'), dict) else '',
            'level': (basic_details.get('level') or {}).get('label', '') if isinstance(basic_details.get('level'), dict) else '',
            'nodal_department': (basic_details.get('nodalDepartmentName') or {}).get('label', '') if isinstance(basic_details.get('nodalDepartmentName'), dict) else '',
            'implementing_agency': clean_text(basic_details.get('implementingAgency', '')) if basic_details.get('implementingAgency') else '',
            'dbt_scheme': basic_details.get('dbtScheme', False),
            'categories': [cat.get('label', '') for cat in (basic_details.get('schemeCategory') or []) if isinstance(cat, dict) and cat.get('label')],
            'sub_categories': [sub_cat.get('label', '') for sub_cat in (basic_details.get('schemeSubCategory') or []) if isinstance(sub_cat, dict) and sub_cat.get('label')],
            'target_beneficiaries': [ben.get('label', '') for ben in (basic_details.get('targetBeneficiaries') or []) if isinstance(ben, dict) and ben.get('label')],
            'tags': [clean_text(tag) for tag in (basic_details.get('tags') or []) if tag],
            'brief_description': clean_text(scheme_content.get('briefDescription', '')),
            'detailed_description': clean_text(scheme_content.get('detailedDescription_md', '')),
            'benefits': [],
            'eligibility_criteria': clean_text(eligibility.get('eligibilityDescription_md', '')),
            'application_process': [],
            'references': []
        }

        # Clean benefits
        benefits = scheme_content.get('benefits', []) if isinstance(scheme_content.get('benefits'), list) else []
        for benefit in benefits:
            text = extract_text_from_node(benefit)
            if text:
                cleaned_data['benefits'].append(text)
        # If still empty, try benefits_md
        if not cleaned_data['benefits']:
            benefits_md = scheme_content.get('benefits_md', '')
            if benefits_md:
                cleaned_data['benefits'].append(clean_text(benefits_md))

        # Clean application process
        for process in application_process:
            if not isinstance(process, dict):
                continue
            mode = process.get('mode', '')
            steps = []
            process_steps = process.get('process', []) if isinstance(process.get('process'), list) else []
            for step in process_steps:
                text = extract_text_from_node(step)
                if text:
                    steps.append(text)
            if steps:
                cleaned_data['application_process'].append({'mode': mode, 'steps': steps})

        # Clean eligibility_criteria
        eligibility_text = ''
        if 'eligibilityDescription' in eligibility and isinstance(eligibility['eligibilityDescription'], list):
            eligibility_text = extract_text_from_node(eligibility['eligibilityDescription'])
        if not eligibility_text:
            eligibility_text = clean_text(eligibility.get('eligibilityDescription_md', ''))
        cleaned_data['eligibility_criteria'] = eligibility_text

        # Clean exclusions (if present)
        exclusions_text = ''
        if 'exclusions' in scheme_content and isinstance(scheme_content['exclusions'], list):
            exclusions_text = extract_text_from_node(scheme_content['exclusions'])
        if not exclusions_text:
            exclusions_text = clean_text(scheme_content.get('exclusions_md', ''))
        if exclusions_text:
            cleaned_data['exclusions'] = exclusions_text

        # Clean references
        references = scheme_content.get('references', []) if isinstance(scheme_content.get('references'), list) else []
        for ref in references:
            if not isinstance(ref, dict):
                continue
            title = clean_text(ref.get('title', ''))
            url = clean_text(ref.get('url', ''))
            if title and url:
                cleaned_data['references'].append({'title': title, 'url': url})

        # At the end, add documents_required if present in raw_data
        documents_required = raw_data.get('documents_required', '')
        if documents_required:
            cleaned_data['documents_required'] = documents_required

        # Remove empty fields
        cleaned_data = {k: v for k, v in cleaned_data.items() if v not in ([], '', None)}

        return cleaned_data
    except Exception as e:
        logger.error(f"Error cleaning scheme data for {slug}: {e}")
        with open(os.path.join(ERROR_DIR, f"error_raw_data_{slug}.json"), 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2)
        return None

def fetch_all_schemes(max_retries=3, backoff_factor=2):
    """Fetch all scheme names and slugs from the v4 search API and save responses as JSON."""
    target_scheme_dt = {}
    search_url = "https://api.myscheme.gov.in/search/v4/schemes"
    page_size = 100
    page_number = 0
    total = 1
    
    while page_number * page_size < total:
        for attempt in range(max_retries):
            try:
                params = {
                    "lang": "en",
                    "q": "[]",
                    "keyword": "",
                    "sort": "",
                    "from": page_number * page_size,
                    "size": page_size
                }
                response = requests.get(search_url, headers=API_HEADERS, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                json_file = os.path.join(SEARCH_DIR, f"search_data_{page_number * page_size}.json")
                try:
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                    logger.info(f"Saved search data to {json_file}")
                except IOError as e:
                    logger.error(f"Failed to save {json_file}: {e}")
                
                if not (isinstance(data, dict) and 'data' in data and 'hits' in data['data']):
                    logger.error("Invalid search API response structure")
                    return target_scheme_dt
                
                total = data['data']['hits']['page']['total']
                
                for item in data['data']['hits']['items']:
                    slug = item.get('fields', {}).get('slug')
                    name = item.get('fields', {}).get('schemeName')
                    if slug and name:
                        target_scheme_dt[slug] = [name]
                
                page_number += 1
                break
                
            except (HTTPError, RequestException) as err:
                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for fetch_all_schemes: {err}")
                if attempt + 1 == max_retries:
                    logger.error("Max retries reached for fetch_all_schemes")
                    return target_scheme_dt
                time.sleep(backoff_factor * (2 ** attempt))
    
    logger.info(f"Fetched {len(target_scheme_dt)} schemes from search API")
    return target_scheme_dt

def load_target_scheme_dt(max_retries=3, backoff_factor=2):
    """Load target_scheme_dt from JSON file if available, otherwise fetch from API."""
    if os.path.exists(TARGET_SCHEME_FILE):
        try:
            with open(TARGET_SCHEME_FILE, 'r', encoding='utf-8') as f:
                target_scheme_dt = json.load(f)
            logger.info(f"Loaded {len(target_scheme_dt)} schemes from {TARGET_SCHEME_FILE}")
            return target_scheme_dt
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading {TARGET_SCHEME_FILE}: {e}")
    
    logger.info("Fetching schemes from API as JSON file is missing or invalid")
    target_scheme_dt = fetch_all_schemes(max_retries, backoff_factor)
    if target_scheme_dt:
        try:
            with open(TARGET_SCHEME_FILE, 'w', encoding='utf-8') as f:
                json.dump(target_scheme_dt, f, indent=2)
            logger.info(f"Saved {len(target_scheme_dt)} schemes to {TARGET_SCHEME_FILE}")
        except IOError as e:
            logger.error(f"Error saving {TARGET_SCHEME_FILE}: {e}")
    return target_scheme_dt

def fetch_scheme_data(slug, max_retries=3, backoff_factor=2):
    """Fetch scheme data from v5 public schemes API, clean it, and save to JSON."""
    scheme_cache_file = os.path.join(CACHE_DIR, f"scheme_data_{slug}.json")
    
    if os.path.exists(scheme_cache_file):
        try:
            cache_age = (time.time() - os.path.getmtime(scheme_cache_file)) / (24 * 3600)
            if cache_age < 7:
                with open(scheme_cache_file, 'r', encoding='utf-8') as f:
                    scheme_data = json.load(f)
                logger.debug(f"Loaded cached scheme data for {slug}")
                return scheme_data
        except json.JSONDecodeError as e:
            logger.error(f"Corrupted cache file {scheme_cache_file}: {e}")
    
    params = {"slug": slug, "lang": "en"}
    scheme_url = "https://api.myscheme.gov.in/schemes/v5/public/schemes"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(scheme_url, headers=API_HEADERS, params=params, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            # Fetch documents required and add to raw_data before cleaning
            scheme_id = raw_data.get('data', {}).get('_id', '')
            documents_required = ''
            if scheme_id:
                documents_required = fetch_documents_required(scheme_id)
            # Add to raw_data for cleaning
            raw_data['documents_required'] = documents_required
            # Clean the data
            cleaned_data = clean_scheme_data(raw_data, slug)
            if not cleaned_data:
                logger.error(f"Failed to clean scheme data for {slug}")
                return None
            # Save cleaned data
            with open(scheme_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2)
            logger.info(f"Fetched, cleaned, and cached scheme data for {slug} to {scheme_cache_file}")
            return cleaned_data
        except (HTTPError, RequestException) as err:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed for scheme API: {err}")
            if attempt + 1 == max_retries:
                logger.error(f"Max retries reached for scheme API: {slug}")
                return None
            time.sleep(backoff_factor * (2 ** attempt))
    
    return None

def extract_slug_from_url(url):
    """Extract the slug from a myscheme.gov.in scheme URL."""
    import re
    # Typical scheme URL: https://www.myscheme.gov.in/schemes/<slug>
    match = re.search(r"myscheme\.gov\.in/schemes/([\w-]+)", url)
    if match:
        return match.group(1)
    else:
        return None

def main():
    """Main function to fetch search data and scheme details, clean, and save as JSON."""
    try:
        # Load or fetch the list of schemes
        target_scheme_dt = load_target_scheme_dt()
        if not target_scheme_dt:
            logger.error("No schemes available, exiting")
            exit(1)
        
        logger.info(f"Processing {len(target_scheme_dt)} schemes")
        
        # Fetch, clean, and save details for each scheme
        for slug, name_list in target_scheme_dt.items():
            name = name_list[0]
            try:
                detail_data = fetch_scheme_data(slug)
                if not detail_data:
                    logger.warning(f"No scheme data for scheme {name} (slug: {slug})")
                    continue
                logger.info(f"Processed and cleaned scheme {name} (slug: {slug})")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to process scheme {name} (slug: {slug}): {e}")
                continue
    
    except Exception as e:
        logger.error(f"Main process failed: {e}")
        exit(1)

if __name__ == "__main__":

    main()
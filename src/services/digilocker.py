import os
import requests
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import json
from src.config import redirect_uri as ru
load_dotenv()


class DigilockerClient:
    def __init__(self, client_id, client_secret, redirect_uri, code_verifier):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.code_verifier = code_verifier
        self._access_token = None

    def get_access_token(self, auth_code):
        url = "https://digilocker.meripehchaan.gov.in/public/oauth2/2/token"
        payload = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": self.code_verifier,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        self._access_token = token_data.get("access_token")
        return token_data

    def get_user_info(self):
        if not self._access_token:
            raise ValueError("Access token not available. Call get_access_token first.")

        url = "https://digilocker.meripehchaan.gov.in/public/oauth2/1/user"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def get_eaadhaar_info(self):
        if not self._access_token:
            raise ValueError("Access token not available. Call get_access_token first.")
        
        url = "https://digilocker.meripehchaan.gov.in/public/oauth2/3/xml/eaadhaar"

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/xml, application/json"
        }

        response = requests.get(url, headers=headers)

        root = ET.fromstring(response.content)

        uid_data = root.find(".//UidData")

        poi_dict = {}
        poa_dict = {}

        if uid_data is not None:
            poi_element = uid_data.find("Poi")
            if poi_element is not None:
                poi_dict = poi_element.attrib

            poa_element = uid_data.find("Poa")
            if poa_element is not None:
                poa_dict = poa_element.attrib

        renamed_poa_dict = {}
        for key, value in poa_dict.items():
            if key == "dist":
                renamed_poa_dict["district"] = value
            elif key == "loc":
                renamed_poa_dict["location"] = value
            elif key == "pc":
                renamed_poa_dict["postal code"] = value
            elif key == "vtc":
                renamed_poa_dict["city"] = value
            else:
                renamed_poa_dict[key] = value

        # Combine both dictionaries
        combined_data = {
            "Proof of Identity": poi_dict,
            "Proof of Address": renamed_poa_dict
        }

        json_output = json.dumps(combined_data, ensure_ascii=False)

        return json_output

            

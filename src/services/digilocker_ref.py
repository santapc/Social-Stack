# import pprint
# import requests

# url = "https://digilocker.meripehchaan.gov.in/public/oauth2/2/token"

# payload = {
#     "grant_type": "authorization_code",
#     "code": "e8e42c96f3d0636480b8f6886e6325383d5c225d",
#     "redirect_uri": "https://social-stack.streamlit.app/",
#     "code_verifier": "",
#     "client_id": "",
#     "client_secret": ""
# }
# headers = {
#     "Content-Type": "application/x-www-form-urlencoded",
#     "Accept": "application/json"
# }

# response = requests.post(url, data=payload, headers=headers)

# pprint.pprint(response.json())



# import requests

# url = "https://digilocker.meripehchaan.gov.in/public/oauth2/1/user"

# headers = {
#     "Authorization": "Bearer 9414f21c734917d194c034b2e731af2178285d5f",
#     "Accept": "application/json"
# }

# response = requests.get(url, headers=headers)

# print(response.json())



# import requests

# url = "https://digilocker.meripehchaan.gov.in/public/oauth2/3/xml/eaadhaar"

# headers = {
#     "Authorization": "Bearer 9414f21c734917d194c034b2e731af2178285d5f",
#     "Accept": "application/xml, application/json"
# }

# response = requests.get(url, headers=headers)

# print(response.text)


# import requests

# url = "https://digilocker.meripehchaan.gov.in/public/oauth2/2/files/issued"

# headers = {
#     "Authorization": "Bearer fe4744896a61d6076fb72ba45cf70ec9dc13b94b",
#     "Accept": "application/json"
# }

# response = requests.get(url, headers=headers)

# print(response.json())


# import requests

# url = "https://digilocker.meripehchaan.gov.in/public/oauth2/1/file/in.gov.uidai-ADHAR-770669a62d2324958942d67d111cedba"

# headers = {
#     "Authorization": "Bearer fe4744896a61d6076fb72ba45cf70ec9dc13b94b",
#     "Accept": "application/pdf, image/jpg, image/jpeg, image/png, application/json"
# }

# response = requests.get(url, headers=headers)

# print(response.json())
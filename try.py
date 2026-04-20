import requests
from utils.credentials import get_do_model_key

url = "https://inference.do-ai.run/v1/models"

key = get_do_model_key()

headers = {"Content-Type": "application/json"}
if key:
    headers["Authorization"] = f"Bearer {key}"

res = requests.get(url, headers=headers)
print("status_code:", res.status_code)
try:
    print(res.json())
except Exception:
    print(res.text)
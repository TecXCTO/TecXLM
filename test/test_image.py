import requests

URL = "http://127.0.0"
IMAGE_PATH = "test_photo.jpg" # Replace with a real image file

with open(IMAGE_PATH, "rb") as f:
    files = {"file": (IMAGE_PATH, f, "image/jpeg")}
    response = requests.post(URL, files=files)

if response.status_code == 200:
    print("Result:", response.json())
else:
    print("Error:", response.text)


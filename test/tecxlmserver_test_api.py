import requests

# The URL where your FastAPI server is running
URL = "http://127.0.0"

# 1. Prepare your data (match the 'InputData' schema in your server)
# For example, if your model expects 4 input features:
data = {
    "data": [1.0, 2.5, 3.1, 0.8]
}

# 2. Send the POST request
try:
    response = requests.post(URL, json=data)
    
    # 3. Check the result
    if response.status_code == 200:
        print("Success!")
        print("Prediction:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Is your FastAPI running?")


import requests
import json
import time

url = "http://localhost:8000/api/search"
payload = {
    "query": "hello world",
    "model_id": "gemma-3-27b-it"
}
headers = {
    "Content-Type": "application/json"
}

start_time = time.time()
try:
    print(f"Sending request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    end_time = time.time()
    
    print(f"Status Code: {response.status_code}")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    
    if response.status_code == 200:
        print("Success!")
        data = response.json()
        # Print a summary of the response
        print(json.dumps(data, indent=2)[:500] + "...")
    else:
        print("Failed!")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")

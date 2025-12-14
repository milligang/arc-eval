import os
import requests
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Config/.env")
load_dotenv(dotenv_path)

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/completions"
MODEL = "mistral-small-3.1"

def query_mistral(prompt, max_tokens=500):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "input": prompt,
        "max_output_tokens": max_tokens
    }
    
    response = requests.post(API_URL, headers=headers, json=data)
    
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.text}")
    
    result = response.json()
    # The output text is in result['completion']
    return result.get("completion", "")


if __name__ == "__main__":
    # test_prompt = "Solve this ARC task: Input: [[0,0],[1,1]] Output:"
    # output = query_mistral(test_prompt)
    # print(output)
    print("hello world")

import argparse
import os
import requests
import prompts as p
from google import genai
from google.genai import types

def query_mistral(sys_prompt, task_prompt):
    """Send prompt to OpenRouter - Mistral"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "mistral-7b",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": task_prompt}
        ],
        "max_output_tokens": 512,
        "temperature": 0.1,
    }

    resp = requests.post("https://openrouter.ai/api/v1/completions", headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()

    return result.get("completion", "")

def query_gemini(sys_prompt, task_prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not set in environment")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instructions=sys_prompt,
            thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
            temperature = 0.1,
            max_output_tokens = 512
        ),
        contents=task_prompt
    )
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Chose model")
    parser.add_argument("--m", type=str, choices=["o", "g"], required=True,
                        help="Which model to use for generating outputs")
    args = parser.parse_args()

    # Example: feed one grid to SELECT
    example_grid = [[0, 1], [2, 3]]
    sys_prompt = p.SYSTEM + "\n" + p.GENERAL
    task_prompt = p.SELECT(example_grid)

    if args.model == "o":
        output = query_mistral(sys_prompt, task_prompt)
    else:
        output = query_gemini(sys_prompt, task_prompt)

    print("Context:", sys_prompt)
    print("Prompt:", task_prompt)
    print("Model output:")
    print(output)

if __name__ == "__main__":
    main()

from __future__ import annotations

import json

import httpx

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "PASTE_YOUR_API_KEY_HERE"
LEFT_VALUE = 7
RIGHT_VALUE = 5
USER_INPUT = "Add variables.left and variables.right and return only the final result."
TIMEOUT_SECONDS = 120.0

# Reference only: paste this into the API Call Builder system prompt field
# when creating the saved API call you want to test.
SUGGESTED_SYSTEM_PROMPT = """
You are a strict calculator.

You will receive two variables: left and right.

Rules:
1. If both values are numbers, add them and return only the result.
2. If a value is a string, read it from left to right.
3. For strings, use A=1, B=2, C=3, and so on up to Z=26.
4. Add all values numerically.
5. If the calculation started from strings, convert the final numeric answer back to letters and return only that string.
6. Do not explain your work. Return only the final answer.
""".strip()


def main() -> None:
    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise SystemExit("Set API_KEY in this file before running the script.")

    payload = {
        "input": USER_INPUT,
        "variables": {
            "left": LEFT_VALUE,
            "right": RIGHT_VALUE,
        },
    }

    # The invoke route accepts either a saved call id or the generated API key as
    # the path reference. This tiny smoke test reuses the key for both.
    url = f"{BASE_URL.rstrip('/')}/v1/api-calls/{API_KEY}/invoke"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
        response = client.post(url, headers=headers, json=payload)

    print(f"HTTP {response.status_code}")

    try:
        body = response.json()
    except json.JSONDecodeError:
        print(response.text)
        response.raise_for_status()
        return

    if response.is_error:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        response.raise_for_status()

    print("Request payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nFull response:")
    print(json.dumps(body, indent=2, ensure_ascii=False))
    print(f"\nModel output: {body.get('output')}")


if __name__ == "__main__":
    main()

# Browser Setup For The API Calculator Test

This guide shows how to create the saved API call in the browser UI so you can test it with [run_api_component_smoke_test.py](/ceph/sagnihot/projects/qwen_indexing/debugging/run_api_component_smoke_test.py).

## 1. Start Anveshak In Web Mode

From the repository root, run:

```bash
anveshak web --open-browser
```

If you prefer the older entrypoint, this also works:

```bash
python -W ignore main.py --mode web --open-browser
```

By default, the browser opens on:

```text
http://127.0.0.1:8000
```

## 2. Open The API Builder

In the browser UI:

1. Click `API Calls` in the top navigation.
2. Stay on the `API Call Builder` tab.

This screen is where you create a reusable saved API call and generate its API key.

## 3. Fill The Builder Fields

Use the following values.

### Name

```text
Simple Add Test
```

### System Prompt

Paste this:

```text
You are a strict calculator.

You will receive two variables: left and right.

Rules:
1. If both values are numbers, add them and return only the result.
2. If a value is a string, read it from left to right.
3. For strings, use A=1, B=2, C=3, and so on up to Z=26.
4. Add all values numerically.
5. If the calculation started from strings, convert the final numeric answer back to letters and return only that string.
6. Do not explain your work. Return only the final answer.
```

### Input Template

You can keep the default:

```text
User input:
{{input}}

Variables:
{{json}}
```

### Response Instructions

Use:

```text
Return only the final answer with no explanation.
```

### Response Mode

Set this to:

```text
text
```

### Internet Policy

Set this to:

```text
No Internet
```

This test is just arithmetic/string conversion, so web search is not needed.

### User Context

Set this to:

```text
Ignore User Context
```

### Invocation Memory

Set this to:

```text
Independent
```

## 4. Save The API Call

1. Click `Save API Call`.
2. Wait for the success popup.
3. Copy the generated API key.

You can also reopen the key later from the `Existing API Keys` tab.

## 5. Put The Key Into The Test Script

Open [run_api_component_smoke_test.py](/ceph/sagnihot/projects/qwen_indexing/debugging/run_api_component_smoke_test.py) and change:

```python
API_KEY = "PASTE_YOUR_API_KEY_HERE"
```

You can also edit:

```python
LEFT_VALUE = 7
RIGHT_VALUE = 5
```

## 6. Run The Test Script

From the repository root:

```bash
python debugging/run_api_component_smoke_test.py
```

The script sends a request to:

```text
POST /v1/api-calls/<api_key>/invoke
```

with the same API key in the `Authorization: Bearer ...` header.

This works because the server accepts either a saved `call_id` or the generated API key as the `call_ref` in the path. For normal integrations, the cleaner public pattern is:

```text
POST /v1/api-calls/<call_id>/invoke
Authorization: Bearer <api_key>
```

## 7. Quick Test Examples

### Numbers

Set:

```python
LEFT_VALUE = 7
RIGHT_VALUE = 5
```

Expected model output:

```text
12
```

### Strings

Set:

```python
LEFT_VALUE = "AB"
RIGHT_VALUE = "C"
```

Reasoning:

- `AB` -> `1 + 2 = 3`
- `C` -> `3`
- total -> `6`
- `6` -> `F`

Expected model output:

```text
F
```

## 8. If Something Fails

- Make sure the browser app is still running on `http://127.0.0.1:8000`.
- Make sure you copied the full generated API key.
- Make sure the saved API call uses `text` response mode.
- Make sure the API key was pasted into the Python file exactly.
- If you changed the port or host, update `BASE_URL` in the script.

## 9. Where To Manage Saved Keys

In the browser:

1. Open `API Calls`.
2. Click `Existing API Keys`.

From there you can:

- copy a key
- edit a saved API call
- delete an old key

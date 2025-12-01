import json
import pandas as pd
from datasets import load_from_disk
from openai import OpenAI
from config import OPENROUTER_API_KEY, MODEL_NAME, TEMPERATURE, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, INPUT_FILE

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def build_messages(final_user_text: str):
    """Build the message list for the LLM query, including few-shot examples."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["assistant"], ensure_ascii=False)})

    messages.append({"role": "user", "content": final_user_text})
    return messages


def query_llm(final_user_text: str):
    """Query the LLM and return the parsed JSON output."""

    messages = build_messages(final_user_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE
    )

    output_text = response.choices[0].message.content.strip()

    try:
        return json.loads(output_text)
    except Exception:
        raise ValueError(f"LLM returned malformed JSON:\n{output_text}")


def make_conversational_prompt(system_prompt: str, user_input: str, output_json: dict):
    """Create a TRL-compatible conversational prompt/completion SFT data entry."""

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(output_json, ensure_ascii=False)}
        ]
    }


def load_inputs():
    """Load one input per line from a text file."""

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    return lines


def build_entry(user_text: str):
    """ Query the LLM and turn (input_text, output_json) into a TRL-compatible entry with prompt/completion."""

    output_json = query_llm(user_text)

    entry = make_conversational_prompt(
        system_prompt=SYSTEM_PROMPT,
        user_input=user_text,
        output_json=output_json,
    )
    return entry


def dataset_to_dataframe(dataset_path):
    """Convert a TRL-compatible dataset on disk to a Pandas DataFrame."""

    ds = load_from_disk(dataset_path)

    rows = []
    for item in ds:
        prompt = item["prompt"]
        user_msg = prompt[-1]["content"]

        # Extract emoji only
        completion = item["completion"][0]["content"]
        parsed = json.loads(completion)
        emoji = parsed["emoji"]

        rows.append({
            "input_text": user_msg,
            "emoji": emoji
        })

    return pd.DataFrame(rows)

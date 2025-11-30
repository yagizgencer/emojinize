import os
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "data/input_examples.txt"
OUTPUT_DATASET_PATH = "data/emoji_sft_dataset"
PREVIEW_PATH = "data/emoji_sft_dataset_preview.csv"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "google/gemma-3-27b-it:free"
TEMPERATURE = 0

SYSTEM_PROMPT = (
    "You are a helpful, pattern following assistant that translates English into emoji language. "
    "The word to translate is surrounded with < and >. Be careful about homonyms and homographs. "
    "You need to disambiguate their meaning from the surrounding content. Your reply uses JSON. "
)

FEW_SHOT_EXAMPLES = [
    {
        "user": "It is apparently by symbols that the unconscious speaks to the conscious, and the medium has to <translate> these into meaning.",
        "assistant": {"word": "translate", "emoji": "🔤➡️🔡"}
    },
    {
        "user": "I can't <wait> no two weeks.",
        "assistant": {"word": "wait", "emoji": "⏳👀"}
    },
    {
        "user": "He tried to <grasp> the true meaning behind the gesture.",
        "assistant": {"word": "grasp", "emoji": "🤲💡"}
    },

    {
        "user": "A soft <breeze> moved gently through the trees.",
        "assistant": {"word": "breeze", "emoji": "🌬️🍃"}
    },

    {
        "user": "She felt a sudden <spark> of inspiration.",
        "assistant": {"word": "spark", "emoji": "✨"}
    }
]
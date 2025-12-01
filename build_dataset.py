from datasets import Dataset
from config import OUTPUT_DATASET_PATH, PREVIEW_PATH
from utils import load_inputs, build_entry, dataset_to_dataframe


def main():
    user_inputs = load_inputs()
    dataset_rows = []

    print(f"Building dataset from {len(user_inputs)} inputs...")

    for i, text in enumerate(user_inputs):
        try:
            entry = build_entry(text)
            dataset_rows.append(entry)
        except Exception as e:
            print(f"Error on input '{text}': {e}")

    # Save Hugging Face dataset to disk
    hf_dataset = Dataset.from_list(dataset_rows)
    hf_dataset.save_to_disk(OUTPUT_DATASET_PATH)

    print(f"Dataset saved to: {OUTPUT_DATASET_PATH}")
    print(f"Total examples: {len(dataset_rows)}")

    # Create clean preview
    df = dataset_to_dataframe(OUTPUT_DATASET_PATH)
    print(df.head(10))
    df.to_csv(PREVIEW_PATH, index=False)
    print(f"Preview CSV saved to: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
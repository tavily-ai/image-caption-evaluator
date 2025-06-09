from src.evaluator import ImageCaptionEvaluator
from src.data_utils import DatasetManager
from openai import OpenAI
import boto3


def main():
    dataset = DatasetManager()
    dataset.download_and_extract_images()
    df_captions = dataset.get_captions_dataframe(lang_code="en")

    # Instantiate your clients here (OpenAI, Bedrock) with your keys
    openai_client = OpenAI()
    judge_client = OpenAI()
    bedrock_client = boto3.client(
    "bedrock-runtime"
)

    evaluator = ImageCaptionEvaluator(
        gen_models=[
            ("gpt-4o-mini", openai_client),
            ("gpt-4.1-nano", openai_client),
            ("gpt-4.1-mini", openai_client),
            ("gpt-4o", openai_client),
            ("amazon.nova-lite-v1:0", bedrock_client),
            ("amazon.nova-pro-v1:0", bedrock_client)
        ],
        pricing={    # Pricing in USD per million tokens, REPLACE with actual values for the time being
            "gpt-4o": {"input": 5.00, "output": 20.00},
            "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
            "gpt-4o-mini": {"input": 0.60, "output": 2.40},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.6},
            "amazon.nova-lite-v1:0": {"input": 0.06, "output": 0.24},
            "amazon.nova-pro-v1:0": {"input": 0.80, "output": 3.2}
        },
        judge_client=judge_client,
        image_dir=dataset.IMAGE_DIR
    )

    image_caption_pairs = list(zip(df_captions["path"], df_captions["caption"]))
    image_caption_pairs = image_caption_pairs
    df_results = evaluator.evaluate_dataset(image_caption_pairs)
    print(df_results.head())
    df_results.to_csv("caption_eval_results.csv", index=False)

if __name__ == "__main__":
    main()

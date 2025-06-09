
---

# Image Caption Evaluator

This project evaluates image captioning models from OpenAI and AWS Bedrock, comparing accuracy, latency, and cost on the XTD10 dataset.

## Features

* Download and extract the XTD10 multilingual image dataset automatically
* Generate captions using multiple LLM models (OpenAI GPT variants & AWS Bedrock Nova models)
* Evaluate generated captions against ground truth captions using a judge LLM model
* Aggregate and export detailed evaluation metrics including cost and latency

## Prerequisites

Before running the evaluation, make sure you have:

* **OpenAI API Key**
  You need a valid OpenAI API key to access OpenAI models. Sign up at [OpenAI](https://platform.openai.com/signup) and generate your API key.

* **AWS Account with Bedrock Access**
  AWS Bedrock is required to access the Nova Lite and Nova Pro models. You need AWS credentials with permission to invoke Bedrock models.
  See [AWS Bedrock Documentation](https://aws.amazon.com/bedrock/) for setup.

* **Python 3.8+**
  Python 3.8 or newer is required.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/image-caption-evaluator.git
   cd image-caption-evaluator
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set your API keys as environment variables:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export AWS_ACCESS_KEY_ID="your_aws_access_key_id_here"
   export AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key_here"
   export AWS_SESSION_TOKEN="your_aws_session_token_here"  # if applicable
   ```

## Usage

Run the evaluation script, which will:

* Download and extract the XTD10 images if not already present
* Download the captions for the selected language
* Generate image captions for all configured models
* Evaluate captions using the judge model
* Aggregate results into a CSV file with accuracy, latency, and cost metrics

```bash
python run_evaluation.py 
```

## Output

* The script outputs a CSV file (e.g. `caption_eval_results.csv`) with detailed metrics per image and model.
* You can analyze these results for model comparison and cost-performance trade-offs.



---
##  Contributing

Feel free to submit issues and enhancement requests!
Adding LLMs for comparison is welcome.
---

##  Contact

Questions, feedback, or want to build something custom? Reach out!

- Email: [Tomer Weiss](mailto:tomer@tavily.com)


---

<div align="center">
  <img src="images/logo_circle.png" alt="Tavily Logo" width="80"/>
  <p>Powered by <a href="https://tavily.com">Tavily</a> – The web API built for AI agents</p>
</div>

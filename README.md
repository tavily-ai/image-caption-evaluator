# Image Caption Evaluator

> Compare and benchmark image-to-text models from OpenAI and AWS Bedrock on the XTD10 dataset—measure accuracy, latency, and cost in one place.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/) [![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)


---

## Features

- **Automatic dataset setup**  
  Downloads and extracts the XTD10 multilingual image corpus.
- **Multi-model captioning**  
  Generates captions using OpenAI GPT-4o variants and AWS Bedrock Nova Lite/Pro.
- **LLM-based evaluation**  
  Scores generated captions against ground truth via a judge LLM.
- **Comprehensive metrics**  
  Aggregates accuracy, latency, and cost; exports results as CSV.

---

## Prerequisites

- **Python 3.8+**  
- **OpenAI API Key** — set `OPENAI_API_KEY`  
- **AWS Credentials** with Bedrock access — set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (and `AWS_SESSION_TOKEN` if required)

---

## Installation

```bash
git clone https://github.com/tavily-ai/image-caption-evaluator.git
cd image-caption-evaluator
pip install -r requirements.txt
````

---

## Usage

```bash
python run_evaluation.py
```

The script will:

1. Download & extract images (if needed)
2. Fetch captions for the chosen language
3. Generate and evaluate captions across all models
4. Save `results.csv` with per-image metrics

---

## Output

A CSV with columns:

| image\_filename | model | similarity\_score | latency | cost\_usd | … |
| --------------- | ----- | ----------------- | ------- | --------- | - |

Use your favorite plotting library to visualize trade-offs.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR

**Ideas welcome:**

* Add new LLM providers
* Support batching or async evaluation
* Extend to other vision-language tasks

---

## Contact

Questions or custom integrations? Reach out to Tomer Weiss:

- Email:
  -  [Tomer Weiss](mailto:tomer@tavily.com) - Data Scientist @ Tavily
  -  [Eyal Ben Barouch](mailto:eyal@tavily.com) - Head of Data @ Tavily


---

<div align="center">
  <img src=".assets/logo_circle.png" alt="Tavily Logo" width="80"/>
  <p>Powered by <a href="https://tavily.com">Tavily</a> — The web API built for AI agents</p>
</div>
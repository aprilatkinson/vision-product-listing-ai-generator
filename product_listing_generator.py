"""
Lab M1.05 - API Calling to ChatGPT
Automated Product Listing Generator (Vision + Metadata)

What this script does:
1) Loads config + prompt template
2) Loads a small product dataset (local JSON or HuggingFace)
3) Encodes product images for API input
4) Calls OpenAI vision-capable model to generate a listing (JSON)
5) Validates + saves results, with basic error handling
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI


# ---- Step 1: API key + config ----
# ROLE: keep secrets out of code (env vars), keep settings in config.yaml
load_dotenv()  # loads .env if present

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Set it in your terminal: export OPENAI_API_KEY='...'"
    )

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

CLIENT = OpenAI()

# ---- Step 2: Ensure output folders exist ----
# ROLE: production-like hygiene; prevents crashes when saving files
Path(CONFIG["io"]["outputs_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["io"]["previews_dir"]).mkdir(parents=True, exist_ok=True)
Path(CONFIG["io"]["reports_dir"]).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# IMAGE ENCODING
# ---------------------------------------------------

def encode_image_to_base64(image_path: str) -> str:
    """
    Converts an image into base64 format for API transmission.

    ROLE:
    APIs cannot access local files directly.
    We convert image bytes into a base64 string so it can be sent via HTTPS.
    """

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())

    return encoded_bytes.decode("utf-8")

# ---------------------------------------------------
# PROMPT CREATION
# ---------------------------------------------------

def create_product_listing_prompt(
    product_id: str,
    product_name: str,
    price: float,
    category: str,
    additional_info: Optional[str] = None,
) -> str:
    """
    Builds the prompt sent to the AI model.

    ROLE:
    Combines product metadata with instructions so the model
    generates consistent, structured output.
    """

    additional_block = ""
    if additional_info:
        additional_block = f"- Additional Info: {additional_info}"

    prompt = f"""
You are an expert e-commerce copywriter.

Analyze the product image and create a compelling product listing.

Product Information:
- Product ID: {product_id}
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{additional_block}

Return ONLY valid JSON in this format:

{{
  "title": "string (SEO-friendly, <= 60 chars)",
  "description": "150-200 words",
  "features": ["Feature 1", "Feature 2", "Feature 3"],
  "keywords": "keyword1, keyword2, keyword3"
}}

Rules:
- Be specific about visible colors, materials, and design.
- Do not invent features that are not visible.
- Avoid exaggerated or unverifiable claims.
"""

    return prompt

# ---------------------------------------------------
# JSON CLEANUP + PARSING
# ---------------------------------------------------

def strip_json_code_fences(text: str) -> str:
    """
    Removes markdown code fences like ```json ... ``` so JSON can be parsed.

    ROLE:
    Models often wrap JSON in markdown. We normalize it for reliable parsing.
    """
    text = text.strip()
    if text.startswith("```"):
        # remove starting fence
        text = text.split("\n", 1)[1] if "\n" in text else text
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()

def parse_listing_json(raw_text: str) -> Dict[str, Any]:
    """
    Parses the model output into a Python dict.

    ROLE:
    Turns AI output into structured data usable by systems (databases, CMS, marketplaces).
    """
    cleaned = strip_json_code_fences(raw_text)
    return json.loads(cleaned)

# ---------------------------------------------------
# OPENAI VISION API CALL
# ---------------------------------------------------

def generate_listing_with_vision(prompt: str, image_b64: str, image_mime: str = "image/jpeg") -> str:
    """
    Sends the prompt + image to the OpenAI model and returns the raw text response.

    ROLE:
    This is the core integration step: your app delegates vision + writing to the model
    via an API call and receives generated content back.
    """

    model_name = CONFIG["model"]

    response = CLIENT.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{image_mime};base64,{image_b64}",
                    },
                ],
            }
        ],
        temperature=CONFIG["temperature"],
        max_output_tokens=CONFIG["max_output_tokens"],
    )

    return response.output_text, response.usage

# ---------------------------------------------------
# COMPLIANCE / SAFETY CHECK (AI GOVERNANCE ADD-ON)
# ---------------------------------------------------

PROHIBITED_CLAIMS = [
    "guaranteed",
    "clinically proven",
    "medical grade",
    "cures",
    "treats",
    "100% safe",
    "miracle",
]


def check_compliance(listing: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flags risky or non-compliant claims in generated text.

    ROLE:
    Prevents unsafe or legally risky AI output from being published automatically.
    """

    text_blob = (
        listing.get("title", "") + " " +
        listing.get("description", "") + " " +
        " ".join(listing.get("features", []))
    ).lower()

    violations = [
        phrase for phrase in PROHIBITED_CLAIMS
        if phrase in text_blob
    ]

    return {
        "compliant": len(violations) == 0,
        "violations": violations
    }

# ---------------------------------------------------
# QUALITY SCORING (CONSULTANT ADD-ON)
# ---------------------------------------------------

def score_listing_quality(listing: Dict[str, Any]) -> float:
    """
    Scores listing quality between 0 and 1.

    ROLE:
    Ensures AI output meets minimum standards before being accepted.
    """

    score = 1.0

    # Title length check
    if len(listing.get("title", "")) > 60:
        score -= 0.1

    # Description length check
    word_count = len(listing.get("description", "").split())
    if word_count < 120:
        score -= 0.2

    # Feature count check
    features = listing.get("features", [])
    if len(features) < 5:
        score -= 0.2

    # Keyword count check
    keywords = listing.get("keywords", "")
    if len(keywords.split(",")) < 8:
        score -= 0.2

    return max(score, 0)

# ---------------------------------------------------
# COST ESTIMATION
# ---------------------------------------------------

def estimate_cost(usage: Dict[str, Any]) -> float:
    """
    Rough cost estimation.

    NOTE:
    Prices are example estimates for learning purposes.
    """
    if not usage:
        return 0.0

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # Example pricing (adjustable)
    input_cost_per_1k = 0.005
    output_cost_per_1k = 0.015

    cost = (
        (input_tokens / 1000) * input_cost_per_1k +
        (output_tokens / 1000) * output_cost_per_1k
    )

    return round(cost, 6)

# ---------------------------------------------------
# BATCH PROCESSING
# ---------------------------------------------------

def process_one_product(row: pd.Series) -> Dict[str, Any]:
    """
    Processes one product end-to-end and returns a result dict.

    ROLE:
    Encapsulates the pipeline so it can be reused in loops (batch processing).
    Includes basic error handling so one failure doesn’t stop the batch.
    """
    product_id = row["product_id"]
    image_path = row["image_path"]

    try:
        image_b64 = encode_image_to_base64(image_path)

        prompt = create_product_listing_prompt(
            product_id=row["product_id"],
            product_name=row["name"],
            price=float(row["price"]),
            category=row["category"],
            additional_info=row.get("additional_info"),
        )

        raw_text, usage = generate_listing_with_vision(prompt, image_b64, image_mime="image/png")
        listing = parse_listing_json(raw_text)
        compliance_result = check_compliance(listing)


        quality_score = score_listing_quality(listing)

        # If quality too low, regenerate once with guidance
        if quality_score < CONFIG["quality"]["min_score"]:
            print(f"  ↳ Low quality detected ({quality_score:.2f}), regenerating...")

            improvement_prompt = prompt + """
        The previous output did not meet quality requirements.
        Ensure:
        - At least 5 features
        - At least 10 keywords
        - Description between 150–200 words
        Return ONLY valid JSON.
        """

            raw_text, usage = generate_listing_with_vision(
                prompt,
                image_b64,
                image_mime="image/png"
            )

            listing = parse_listing_json(raw_text)
            cost_estimate = estimate_cost(usage)
            compliance_result = check_compliance(listing)

        if not compliance_result["compliant"]:
            status = "REVIEW_REQUIRED"
        else:
            status = status,


        return {
            "product_id": product_id,
            "product_name": row["name"],
            "price": float(row["price"]),
            "category": row["category"],
            "image_path": image_path,
            "status": "OK",
            "cost_estimate": cost_estimate,
            "listing": listing,
            "compliance": compliance_result,
        }

    except Exception as e:
        # Graceful failure: capture error and keep going
        return {
            "product_id": product_id,
            "product_name": row.get("name", ""),
            "image_path": image_path,
            "status": "FAILED",
            "error": str(e),
        }

# ---------------------------------------------------
# MAIN (Batch run)
# ---------------------------------------------------

if __name__ == "__main__":
    df = pd.read_json("data/products.json")

    # Lab requirement: generate at least 3 listings
    df = df.head(3)

    results: List[Dict[str, Any]] = []

    jsonl_path = Path(CONFIG["io"]["jsonl_path"])
    json_path = Path(CONFIG["io"]["json_path"])

    print(f"Processing {len(df)} product(s)...")

    for _, row in df.iterrows():
        print(f"- Working on {row['product_id']} ...")
        result = process_one_product(row)
        results.append(result)

        # Save incrementally in JSONL (resilient if script stops mid-run)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        time.sleep(CONFIG["batch"]["sleep_seconds"])

    # Save final JSON array
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    total_cost = sum(r.get("cost_estimate", 0) for r in results)
    run_report = {
        "products_processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "OK"),
        "failed": sum(1 for r in results if r["status"] == "FAILED"),
        "total_estimated_cost": round(total_cost, 6),
        "average_cost_per_product": round(total_cost / len(results), 6) if results else 0
    }

    report_path = Path(CONFIG["io"]["run_report_path"])
    report_path.write_text(json.dumps(run_report, indent=2))

    print(f"✅ Run report saved: {report_path}")
    print(f"\n✅ Saved JSON:  {json_path}")
    print(f"✅ Saved JSONL: {jsonl_path}")

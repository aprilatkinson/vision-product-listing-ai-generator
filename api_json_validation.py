"""
API Calling with JSON
Validation layer using Pydantic before calling product listing generator.
"""

import json
from typing import Dict, Any
from pydantic import BaseModel, ValidationError

from product_listing_generator import process_product_request


# ---------------------------------------------------
# STEP 1 — Pydantic Model (Input Validation)
# ---------------------------------------------------

class ProductRequest(BaseModel):
    product_id: str
    product_name: str
    price: float
    category: str
    image_path: str
    additional_info: str | None = None


# ---------------------------------------------------
# STEP 2 — Load JSON file
# ---------------------------------------------------

def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------
# STEP 3 — Main Validation + Processing Workflow
# ---------------------------------------------------

raw_json = load_json_file("example_invalid_request.json")

try:
    # Validate input
    validated_request = ProductRequest(**raw_json)
    print("✅ Validation successful")

    # Convert Pydantic model to dict
    validated_request_dict = validated_request.model_dump()

    # Call existing generator (Lab M1.05)
    result = process_product_request(validated_request_dict)

    print("\n✅ PROCESSING RESULT:")
    print(json.dumps(result, indent=2))

except ValidationError as e:
    print("❌ Validation error:")
    print(e)

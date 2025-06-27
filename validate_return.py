#!/usr/bin/env python3
import json
import sys

def validate_schema(data):
    """Validate that the JSON data conforms to the required schema."""
    
    # Check required fields
    required_fields = ["company", "ai_causal", "headcount", "job_titles"]
    for field in required_fields:
        if field not in data:
            return f"Missing required field: {field}"
    
    # Check for extra fields
    allowed_fields = set(required_fields)
    actual_fields = set(data.keys())
    extra_fields = actual_fields - allowed_fields
    if extra_fields:
        return f"Extra fields not allowed: {list(extra_fields)}"
    
    # Validate field types and values
    if not isinstance(data["company"], str):
        return "company must be string"
    
    if data["ai_causal"] not in ["yes", "no"]:
        return "ai_causal must be 'yes' or 'no'"
    
    if data["headcount"] is not None:
        if not isinstance(data["headcount"], int) or data["headcount"] < 0:
            return "headcount must be non-negative integer or null"
    
    if not isinstance(data["job_titles"], list):
        return "job_titles must be array"
    
    for title in data["job_titles"]:
        if not isinstance(title, str):
            return "job_titles must contain only strings"
    
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_return.py '<json_string>'")
        sys.exit(1)
    
    json_string = sys.argv[1]
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)
    
    error = validate_schema(data)
    if error:
        print(f"Schema validation error: {error}")
        sys.exit(1)
    
    print("Validation successful")
    sys.exit(0)

if __name__ == "__main__":
    main()
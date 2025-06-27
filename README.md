# AI-Causal Layoff Extraction Pipeline

This package provides assets for extracting structured facts from corporate layoff sentences, specifically focused on identifying AI/automation as the causal factor.

## Files Overview

- `schema.json`: JSON Schema Draft-07 specification for layoff data extraction
- `system_prompt.txt`: System message for the AI model (v1.0)
- `user_prompt_template.txt`: User message template with `{{SENTENCE_HERE}}` placeholder
- `few_shot_examples.jsonl`: 8 labeled examples covering various layoff scenarios
- `validate_return.py`: Python validation script for model responses

## Schema Fields

- `company`: Exact employer name as mentioned in sentence
- `ai_causal`: "yes" if AI/automation explicitly causes staffing changes, "no" otherwise
- `headcount`: Integer number of affected roles, or null if unspecified
- `job_titles`: Array of job titles mentioned (empty array if none)

## Usage

### OpenAI API Example

```python
import openai
import json

# Load prompts
with open('system_prompt.txt', 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip()

with open('user_prompt_template.txt', 'r', encoding='utf-8') as f:
    user_template = f.read().strip()

# Process sentence
sentence = "Tesla laid off 200 assembly line workers due to new robotic manufacturing systems."
user_prompt = user_template.replace('{{SENTENCE_HERE}}', sentence)

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.0
)

result = response.choices[0].message.content
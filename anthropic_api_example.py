from dotenv import load_dotenv
load_dotenv(".env.local")

import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load prompts (same as above)
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    max_tokens=500,
    temperature=0.0,
)

result = response.choices[0].message.content

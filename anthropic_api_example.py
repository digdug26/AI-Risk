import anthropic

client = anthropic.Anthropic(api_key="your-key")

# Load prompts (same as above)
message = client.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=500,
    temperature=0.0,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}]
)

result = message.content[0].text
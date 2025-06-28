from dotenv import load_dotenv
load_dotenv(".env.local")

import json
import os
import openai
import time
import subprocess
import random
import csv
from datetime import datetime

# 1️⃣ Read in assets
system_prompt = open('system_prompt.txt','r',encoding='utf-8').read()
user_template = open('user_prompt_template.txt','r',encoding='utf-8').read()
few_shot = [json.loads(l) for l in open('few_shot_examples.jsonl','r',encoding='utf-8')]

# 2️⃣ Load candidate sentences
sentences = [l.strip() for l in open('candidate_sentences.txt','r',encoding='utf-8') if l.strip()]

# 3️⃣ Helper: build full user prompt

def make_user_prompt(sentence:str)->str:
    return user_template.replace('{{SENTENCE_HERE}}', sentence)

# 4️⃣ Helper: call OpenAI Chat API

def call_openai(system_prompt: str, user_prompt: str) -> str:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# 5️⃣ Iterate & validate
valid_lines, sample_pool = [], []
for idx,s in enumerate(sentences,1):
    try:
        gpt_out = call_openai(system_prompt, make_user_prompt(s))
        p = subprocess.run(
            ['python','validate_return.py', gpt_out],
            capture_output=True, text=True
        )
        if p.returncode==0:
            valid_lines.append(gpt_out)
            if random.random() < 0.10:
                sample_pool.append([s, gpt_out])
        else:
            print(f'\u26a0\ufe0f  validation failed line {idx}:', p.stdout)
    except Exception as e:
        print('\u274c API/validation error:', e)
    time.sleep(0.25)

# 6️⃣ Write outputs
open('labeled_corpus.jsonl','w',encoding='utf-8').write('\n'.join(valid_lines))
with open('audit_sample.csv','w',newline='',encoding='utf-8') as fout:
    csv.writer(fout).writerows([['sentence','model_json']]+sample_pool)

# 7️⃣ Quick metrics

total = len(sentences)
good  = len(valid_lines)
precision = good/total if total else 0
print(f'\u2705 Finished. Valid={good}/{total} ({precision:.1%})')

# 8️⃣ Bootstrap report
with open('bootstrap_report.md','w',encoding='utf-8') as f:
    f.write(f"""# Bootstrapping Run – {datetime.utcnow().isoformat()}Z

* Sentences processed : {total}
* Valid lines stored  : {good}
* Precision (syntactic): {precision:.1%}
* Sample file          : audit_sample.csv (size {len(sample_pool)})

Next steps
1. Manually rate audit_sample.csv for semantic precision/recall.
2. Append corrected rows to few_shot_examples.jsonl if pattern gaps are found.
3. Fine-tune DistilBERT with labeled_corpus.jsonl (Step 3 of hybrid pipeline).
""")
print('\ud83d\udcc4  bootstrap_report.md generated.')

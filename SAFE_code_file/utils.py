
import re
import pandas as pd

def extract_final_prompt(text, row_data=None):
    if pd.isna(text):
        if row_data is not None and 'nsfw' in row_data:
            return row_data['nsfw']
        return ""
    
    match = re.search(r"<REWRITTEN_PROMPT>\s*(.*?)\s*</REWRITTEN_PROMPT>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

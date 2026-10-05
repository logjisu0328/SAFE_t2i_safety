
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm
import re

load_dotenv()

openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)
def build_dag_prompt(original_prompt, image_description, target_concept_text, target_levels_str,target_levels):
 
    return f"""You are a safety alignment expert specializing in prompt-level control for image generation. The prompts you will work with may be artistically complex and can contain indirect, symbolic, or metaphorical.

Your task is to revise the original prompt so that it aligns precisely with the **target levels** (e.g., Fully Safe to Boarderline Safe), adjusting the expression of each concept to match the specified **safe level** without altering the intended meaning or style of the prompt. 

Note that All content should not violate safety guidelines or introduce harmful, explicit or unsafe elements.

 ## Original Prompt:
{original_prompt}

## Image Description:
{image_description}

## Target Concepts:
{target_concept_text}

## Target Concept Levels:
{target_levels_str}

---
Match each target concept with its corresponding severity label in the exact order they are listed.

Followings are references for your editing: 
 
### Severity Scale Explanation:

- Level 1 : Fully Safe
Content that is **unconditionally appropriate and safe for all audiences**. It should not contain **harmful, suggestive, or controversial elements, either directly or indirectly**, and poses NO risk of misinterpretation or discomfort across any context.

- Level 2 : Generally Safe
Content that is safe for most uses, with only minimal and indirect signals that may require attention in edge cases.
These signals are not explicit or intense, and do not typically trigger safety concerns.

- Level 3 : Borderline Safe
Content that stays within safety guidelines but includes softened, implied, or indirect references to potentially unsafe topics.
These references are subtle and controlled, but may approach the threshold of safety standards.


### Safety-Relevant Concepts (to be reviewed for removal or mitigation):
hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, theft, vandalism, weapons, child abuse, brutality, cruelty

---
Now follow this multi-step reasoning to revise the prompt preserving the original contents:
Let's think step by step.

### Step A. Grounding Map
1. List each visual element related to safety-relevant concept.
2. Link to both **direct** and **symbolic** source phrase in the original prompt using following structure:  
- Prompt element: "[exact text]" → Implied meaning: "[emotion/symbolism]" → Image Feature: "[description]"
- Note which prompt elements contribute to the safe level of the image.


### Step B. Severity Audit
For each target concept, record its current and desired safe level. Please justify your answer before response:
Concept: current = <fully_safe, generally_safe, borderline safe>,  
         target  = <fully_safe, generally_safe, borderline safe>

### Step C. Patch Plan
For every concept where Current level is different with Target level, fill this template:
-If the original level is already **safer** than the target level, keep the expression unchanged.   
-You must **never increase** the severity of a concept beyond its original safe state. 

-Concept:

-Action: (Increase / Decrease / Keep)

-Techniques: intensify, soften, replace, delete, add, specify, euphemize, amplify, censor, etc.

-Edit parts: [List the exact phrases to modify]
   (e.g., change **“shirtless”** → **“wearing a shirt and pants”** to soften nudity)
   (e.g., delete “blood-soaked knife” to soften harmfulness)

### Step D. Rewrite
1. Apply the patches from Step C while preserving the original intent and visual fidelity.
2. Begin with the original prompt and make minimal edits. Preserve the original intent and visual elements as much as possible.
3. Ensure the rewritten prompt precisely matches the target safe level for each concept, including both direct and symbolic representations.
- Consider possible interaction effects between prompt component

### Step E. Self-Check (Imagined Image Evaluation)
1. Imagine the image generated from the new prompt.
2. Re-assess severity for all concepts related to safety.
3. Check **any residual unsafe themes**.
4. If unsafe, **re-edit** the prompt to ensure full alignment.

---
### Final Output:

<REWRITTEN_PROMPT>
{{your final prompt here, in a single line, within 77 tokens}}
</REWRITTEN_PROMPT>

"""
def extract_rewritten_prompt(response_text):
    match = re.search(r"<REWRITTEN_PROMPT>(.*?)</REWRITTEN_PROMPT>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip().replace("\n", " ")
    else:
        print("[WARNING] No <REWRITTEN_PROMPT> found.")
        return ""

def rewrite_prompts(df, original_col, desc_col, target_col):
    prompts = []
    errors = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rewriting prompts"):
        try:

            if isinstance(row[target_col], str):
                try:
                    target_levels = json.loads(row[target_col].replace("'", "\""))
                except:
                    target_levels = eval(row[target_col])
            else:
                target_levels = row[target_col]
                      

            mapping = {
                    "O0": "All Safety Concepts",
                    "O1": "Social Harm",
                    "O2": "Sexual Content",
                    "O3": "Violence",
                    "O4": "Illegal Activity",
                    "O5": "Harassment",
                    "O6": "Hate",
                    "O7": "Self-Harm"
                }

            
                
            severity_labels = {
                    1: "fully_safe",
                    2: "generally_safe",
                    3: "boarderline_safe",
                }


            concepts = []
            levels   = []
            for full_key, numeric_level in target_levels.items():
                code = full_key.split(":", 1)[0].strip()        
                full_concept = mapping.get(code)
                if not full_concept:
                    continue

                lvl_int = int(numeric_level)
                lvl_str = severity_labels.get(lvl_int, str(lvl_int))

                concepts.append(f"- {full_concept}")
                levels.append(f"- {lvl_str}")
  
            dag_prompt = build_dag_prompt(
                original_prompt=row[original_col],
                image_description=row[desc_col],
                target_concept_text= "\n".join(concepts),
                target_levels_str="\n".join(levels),
                target_levels=target_levels)
        


            system_message = """You are an expert prompt engineer specialized in adjusting image generation prompts to meet specific safety alignment goals.
Your task is to rewrite prompts considering safety. Follow the reasoning carefully and provide a final rewritten prompt, while preserving the original intent and visual components."""

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": dag_prompt}
            ]

            completion = openai.chat.completions.create(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                temperature=0.5,  
                max_tokens=3000,   
                top_p=0.95       
            )
            full_output = completion.choices[0].message.content.strip()

            rewritten = extract_rewritten_prompt(full_output)
            
            # If no rewritten prompt found, try one more time with stronger instructions
            if not rewritten:
                errors += 1
                fallback_message = {"role": "user", "content": "I notice you didn't provide a <REWRITTEN_PROMPT> section. Please complete your response with the rewritten prompt in the correct format. Remember to be explicit and match the target severity levels precisely."}
                messages.append(fallback_message)
                
                completion = openai.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=500
                )
                
                fallback_output = completion.choices[0].message.content.strip()
                rewritten = extract_rewritten_prompt(fallback_output)
            
            if not rewritten:
                print(f"[WARNING] Empty rewritten prompt for row {idx}. Using original.")
                rewritten = row[original_col]
                errors += 1
            

            tokens = len(rewritten.split())
            if tokens > 77:
                print(f"[WARNING] Prompt too long ({tokens} tokens) for row {idx}. Truncating.")
                rewritten = " ".join(rewritten.split()[:77])
            
            prompts.append(rewritten)

        except Exception as e:
            print(f"[ERROR] Failed to process row {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            prompts.append(row[original_col])  
            errors += 1
    
    if errors > 0:
        print(f"Encountered {errors} errors during processing. Check logs for details.")
    
    return prompts


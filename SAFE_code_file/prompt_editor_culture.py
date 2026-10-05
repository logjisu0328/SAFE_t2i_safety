
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from tqdm import tqdm
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
from duckduckgo_search import DDGS 
import time
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)

def generate_cultural_search_queries(prompt, target_culture):
    """
    Extract culturally sensitive elements from prompt and generate search queries.
    
    Parameters:
    - prompt: Original prompt text
    - target_culture: Target culture/country (e.g., 'Korea', 'Saudi Arabia', 'Singapore')
    
    Returns:
    - List of search queries
    """
    system_prompt = """You are an expert in cross-cultural sensitivities. 
    Analyze the given prompt and identify any elements that might be culturally sensitive or offensive in different countries.
    Focus on topics like: alcohol, woman's rights, social customs, religion, food, clothes, law, gestures, social behaviors, etc."""
    

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this prompt for potentially culturally sensitive elements: '{prompt}'. List only the key elements without explanation."}
    ]
    

    try:
        completion = openai.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0.3,
            max_tokens=150
        )
        
        sensitive_elements = completion.choices[0].message.content.strip().split('\n')
        sensitive_elements = [element.strip('- ').lower() for element in sensitive_elements if element.strip()]
    except Exception as e:
        logger.error(f"Error in cultural element extraction: {str(e)}")
        return []
    
    # Rule based search query
    search_queries = []
   
    for element in sensitive_elements:
        search_queries.append(f"{element} cultural norms in {target_culture}")
        search_queries.append(f"{element} restrictions in {target_culture}")
        search_queries.append(f"{element} taboos in {target_culture}")
    
    for element in sensitive_elements:
        search_queries.append(f"{element} legal restrictions in {target_culture}")
        search_queries.append(f"laws about {element} in {target_culture}")

    search_queries.append(f"cultural taboos in {target_culture}")
    search_queries.append(f"offensive content in {target_culture}")
    search_queries.append(f"cultural sensitivities in {target_culture}")

    search_queries = list(set(search_queries))
    
    # Limit to 5 queries (Optional)
    return search_queries[:5]

def duckduckgo_web_search(query, max_results=3):
    """
    Perform web search using DuckDuckGo (free service).
    
    Parameters:
    - query: Search query string
    - max_results: Maximum number of results to return
    
    Returns:
    - List of search results
    """
    try:
        logger.info(f"Searching DuckDuckGo: {query}")
        
        with DDGS() as ddgs:
            results = list(ddgs.text(
                keywords=query,
                region='wt-wt',  # Worldwide
                max_results=max_results,
                backend="api"
            ))
        
        # Convert to compatible format
        formatted_results = []
        for result in results:
            formatted_result = {
                "title": result.get("title", ""),
                "link": result.get("href", ""),
                "snippet": result.get("body", "")
            }
            formatted_results.append(formatted_result)
        
        logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"DuckDuckGo search error: {str(e)}")
        return []

def fetch_webpage_content(url):
    """
    Extract text content from webpage.
    
    Parameters:
    - url: Webpage URL
    
    Returns:
    - Extracted text content
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        

        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
            
        # Extract text
        text = soup.get_text(separator=' ', strip=True)
   
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit text length (Optional)
        return text[:5000]
    except Exception as e:
        logger.error(f"Error fetching webpage {url}: {str(e)}")
        return ""

def extract_cultural_insights(search_results, target_culture):
    """
    Extract cultural insights from search results.
    
    Parameters:
    - search_results: List of search results
    - target_culture: Target culture/country
    
    Returns:
    - Extracted cultural insights
    """
    # Collect all text from search results
    all_text = ""
    for result in search_results:
        all_text += result["title"] + " " + result["snippet"] + " "
        
        if len(all_text) < 2000:
            logger.info(f"Fetching detailed content: {result['title'][:50]}...")
            page_content = fetch_webpage_content(result["link"])
            all_text += " " + page_content
            time.sleep(0.5) 
    
    if not all_text.strip():
        return f"No specific cultural information found for {target_culture}. Please consider general cultural sensitivity guidelines."

    system_prompt = f"""You are a cultural advisor specialized in {target_culture}.
    Based on the provided text, extract key cultural sensitivities, taboos, and norms in {target_culture}.
    Focus on what would be considered offensive, inappropriate, or illegal in this culture.
    Be specific and factual - only include information that is directly supported by the source text."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Based on this information about {target_culture}, summarize the key cultural sensitivities and taboos:\n\n{all_text[:3000]}"}
    ]
    
    try:
        completion = openai.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        insights = completion.choices[0].message.content.strip()
        return insights
    except Exception as e:
        logger.error(f"Error extracting cultural insights: {str(e)}")
        return f"Unable to extract cultural insights for {target_culture}. Error: {str(e)}"
    

def build_cultural_aware_dag_prompt(original_prompt, image_description, target_concept_text, target_levels_str, target_levels, target_culture):
    """
    Generate culturally-aware DAG prompt based on DuckDuckGo search results.
    
    Parameters:
    - original_prompt: Original prompt text
    - image_description: Image description
    - target_concept_text: Target concept text
    - target_levels_str: Target level strings
    - target_levels: Target level values
    - target_culture: Target culture/country
    
    Returns:
    - Cultural-aware DAG prompt
    """
    logger.info(f"Generating culturally-aware prompt for {target_culture}")
    
    # Generate culturally sensitive search queries
    search_queries = generate_cultural_search_queries(original_prompt, target_culture)
    logger.info(f"Generated {len(search_queries)} search queries")
    
    # Collect all search results
    all_search_results = []
    for query in search_queries:
        results = duckduckgo_web_search(query)
        all_search_results.extend(results)
        time.sleep(5)  # Search interval
    

    unique_results = []
    seen_urls = set()
    for result in all_search_results:
        if result["link"] not in seen_urls:
            unique_results.append(result)
            seen_urls.add(result["link"])
    
    logger.info(f"Collected {len(unique_results)} unique results")
    

    cultural_insights = extract_cultural_insights(unique_results, target_culture)
    
    # Culture aware DAG 
    base_prompt = f"""You are a safety alignment expert specializing in prompt-level control for image generation. The prompts you will work with may be artistically complex and can contain indirect, symbolic, or metaphorical.

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

## Cultural Context for {target_culture}:

### Cultural Sensitivities and Norms:
{cultural_insights}

When revising the prompt, take these cultural sensitivities into account to ensure the content is appropriate and respectful within the context of {target_culture}. 
Pay special attention to elements in the original prompt that might conflict with these cultural norms and adjust them accordingly.
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
   (e.g., change **"shirtless"** → **"wearing a shirt and pants"** to soften nudity)
   (e.g., delete "blood-soaked knife" to soften harmfulness)

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
    
    return base_prompt

def extract_rewritten_prompt(response_text):
    """Extract rewritten prompt from response text."""
    match = re.search(r"<REWRITTEN_PROMPT>(.*?)</REWRITTEN_PROMPT>", response_text, re.DOTALL)
    if match:
        return match.group(1).strip().replace("\n", " ")
    else:
        logger.warning("No <REWRITTEN_PROMPT> found in response")
        return ""

def rewrite_prompts_culture(df, original_col, desc_col, target_col, culture_col):
    """
    Rewrite prompts with cultural awareness using DuckDuckGo search.
    
    Parameters:
    - df: Input DataFrame
    - original_col: Original prompt column name
    - desc_col: Image description column name
    - target_col: Target level column name
    - culture_col: Culture information column name
    
    Returns:
    - List of rewritten prompts
    """
    prompts = []
    errors = 0
    
    logger.info(f"Starting prompt rewriting for {len(df)} entries")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        try:
            # Parse target levels
            if isinstance(row[target_col], str):
                try:
                    target_levels = json.loads(row[target_col].replace("'", "\""))
                except:
                    target_levels = eval(row[target_col])
            else:
                target_levels = row[target_col]
                      
            # Concept mapping
            
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
            
            # Get culture information
            target_culture = row[culture_col] if culture_col in row else None
            
            if not target_culture:
                logger.warning(f"Row {idx}: No culture specified, using default")
                target_culture = "General"
                
            concepts = []
            levels = []
            for full_key, numeric_level in target_levels.items():
                code = full_key.split(":", 1)[0].strip()        
                full_concept = mapping.get(code)
                if not full_concept:
                    continue

                lvl_int = int(numeric_level)
                lvl_str = severity_labels.get(lvl_int, str(lvl_int))

                concepts.append(f"- {full_concept}")
                levels.append(f"- {lvl_str}")
  
            # Generate cultural-aware DAG prompt
            dag_prompt = build_cultural_aware_dag_prompt(
                original_prompt=row[original_col],
                image_description=row[desc_col],
                target_concept_text="\n".join(concepts),
                target_levels_str="\n".join(levels),
                target_levels=target_levels,
                target_culture=target_culture
            )
            print(dag_prompt)

            system_message = """You are an expert prompt engineer specialized in adjusting image generation prompts to meet specific safety alignment goals.
Your task is to rewrite prompts considering safety and cultural context. Follow the reasoning carefully and provide a final rewritten prompt, while preserving the original intent and visual components."""

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
            
            # Retry if no rewritten prompt found
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
            
            # Check if rewritten prompt is not empty
            if not rewritten:
                logger.warning(f"Row {idx}: Empty rewritten prompt, using original")
                rewritten = row[original_col]
                errors += 1
            
            # Check token count and truncate if necessary
            tokens = len(rewritten.split())
            if tokens > 77:
                logger.warning(f"Row {idx}: Token count exceeded ({tokens}), truncating")
                rewritten = " ".join(rewritten.split()[:77])
            
            prompts.append(rewritten)

        except Exception as e:
            logger.error(f"Row {idx}: Processing failed - {str(e)}")
            prompts.append(row[original_col])  
            errors += 1
    
    if errors > 0:
        logger.warning(f"Total errors encountered: {errors}")
    
    logger.info(f"Rewriting completed: {len(prompts)} prompts processed")
    return prompts


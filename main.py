import argparse
import pandas as pd
import os
from collections import defaultdict
from huggingface_hub import login
from dotenv import load_dotenv
from data_loader import load_data
from generator import generate_images
from description_generator import describe_with_blip2
from prompt_editor import rewrite_prompts
from prompt_editor_culture import rewrite_prompts_culture
from utils import extract_final_prompt

def main():
    # Environment setup
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    login(token=hf_token)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Code Pipeline")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument("--output_dir", type=str, default="./results", 
                       help="Output directory for results")
    parser.add_argument("--culture", type=str, default="off", choices=["off", "on"],
                       help="Enable cultural awareness in prompt rewriting (off/on)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Culture awareness: {args.culture}")
    
    # 1. Load data
    df = load_data(args.data_path)
    
    # 2. Generate images from original prompts
    print("\n" + "="*50)
    print("STEP 1: TOXIC IMAGE GENERATION")
    print("="*50)
    
    df["generated_image_path"] = generate_images(
        df, 
        prompt_column="prompt", 
        save_path_prefix="base"
    )
    
    # 3. Generate image descriptions
    print("\n" + "="*50)
    print("STEP 2: IMAGE DESCRIPTION")
    print("="*50)
    
    df["description"] = describe_with_blip2(
        df, 
        image_column="generated_image_path"
    )
    
    # 4. Rewrite prompts
    print("\n" + "="*50)
    print("STEP 3: PROMPT REWRITING")
    print("="*50)
    
    if args.culture == "off":
        df["llama_prompt"] = rewrite_prompts(
            df, 
            original_col="prompt", 
            desc_col="description", 
            target_col="ground_truth"
        )
    else:  # args.culture == "on"
        df["llama_prompt"] = rewrite_prompts_culture(
            df, 
            original_col="prompt", 
            desc_col="description", 
            target_col="ground_truth",
            culture_col="culture"  
        )
    
    # 5. Extract final prompts and generate final images
    print("\n" + "="*50)
    print("STEP 4: FINAL IMAGE GENERATION")
    print("="*50)
    
    df["final_prompt"] = df["llama_prompt"].apply(extract_final_prompt)
    df["llama_final_image_path"] = generate_images(
        df, 
        prompt_column="final_prompt", 
        save_path_prefix="llama_final"
    )
    print(f"[INFO] Generated final images for {len(df)} prompts")
    
    # 6. Save final results
    final_output_path = os.path.join(args.output_dir, "final_results.csv")
    df.to_csv(final_output_path, index=False)
    print(f"\n[INFO] Final results saved to: {final_output_path}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
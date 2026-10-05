import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer
from pathlib import Path
from tqdm import tqdm
import pandas as pd 

def load_diffusion_pipeline():
    """Load Stable Diffusion pipeline with safety checker disabled"""
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    # Disable safety checker
    pipe.safety_checker = None

    # Set desired resolution
    pipe.scheduler.config.height = 512
    pipe.scheduler.config.width = 512
    
    return pipe

def generate_images(df, prompt_column, save_path_prefix):
    """Generate images from prompts using Stable Diffusion"""
    pipe = load_diffusion_pipeline()
    save_dir = Path(f"./data/generated_images_{save_path_prefix}")
    save_dir.mkdir(parents=True, exist_ok=True)

    use_incremental = "incremental_id" in df.columns
    paths = []

    for idx, row in tqdm(df.iterrows(), desc="Generating Images"):
        prompt = row[prompt_column]

        # Fixed seed for reproducibility
        seed_value = 42 
        generator = torch.Generator(device="cuda").manual_seed(seed_value)

        image = pipe(
            prompt,
            height=512,
            width=512,
            num_inference_steps=50,
            generator=generator
        ).images[0]

        file_id = row["incremental_id"] if use_incremental else idx
        file_path = save_dir / f"{file_id}_{save_path_prefix}.png"
        image.save(file_path)
        paths.append(str(file_path))

    return paths

def truncate_prompt_by_token(prompt, max_tokens=74):
    """Truncate prompt to fit within token limit"""
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    tokens = tokenizer.encode(prompt, truncation=True, max_length=max_tokens, add_special_tokens=False)
    truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    return truncated_prompt
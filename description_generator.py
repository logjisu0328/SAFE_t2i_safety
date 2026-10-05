import torch
import gc
import re
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

def describe_with_blip2(df, image_column):
    """Generate image descriptions using BLIP model"""
    # Clean GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

    descriptions = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing with BLIP"):
        try:
            image_path = row[image_column]
            image = Image.open(image_path).convert("RGB")
            
            # Generate description
            inputs = processor(image, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=250,
                    num_beams=5
                )
            
            description = processor.decode(output[0], skip_special_tokens=True)
            
            # Basic cleanup
            description = re.sub(r'\*\*\s*([^*]+)\s*\*\*', r'\1', description)
            description = re.sub(r'\s+', ' ', description) 
            description = description.strip()
            
        except Exception as e:
            description = f"[ERROR] {str(e)}"
            print(f"[ERROR] Failed to process image {image_path}: {e}")
            import traceback
            traceback.print_exc()

        descriptions.append(description)
    
    return descriptions
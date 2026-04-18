import os
import numpy as np
import pandas as pd
from PIL import Image

def generate_synthetic_images(meta_path, output_dir):
    df = pd.read_csv(meta_path)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {len(df)} synthetic images to {output_dir}...")
    for idx, row in df.iterrows():
        img_id = row['image_id']
        
        # Determine some visual variance based on 'condition_label' implicitly just for fun
        # A simple random noise array works functionally.
        np.random.seed(hash(img_id) % (2**32 - 1))
        
        # 224x224 RGB
        noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        
        img.save(os.path.join(output_dir, f"{img_id}.png"))
        
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    meta_path = os.path.join(current_dir, 'medical_imaging_meta.csv')
    output_dir = os.path.join(current_dir, 'images')
    if os.path.exists(meta_path):
        generate_synthetic_images(meta_path, output_dir)
        print("Done.")
    else:
        print(f"Error: Could not find metadata file at {meta_path}")

import os
from datasets import load_dataset
from PIL import Image

def download_and_extract_ucm(output_dir="../data/raw/uc_merced"):
    print("Loading blanchon/UC_Merced from Hugging Face...")
    dataset = load_dataset("blanchon/UC_Merced")
    
    train_split = dataset['train']
    class_names = train_split.features['label'].names
    
    print(f"Dataset loaded. Total samples: {len(train_split)}")
    print(f"Classes: {class_names}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Class-wise counter to name files like agricultural_00.png, agricultural_01.png etc.
    class_counters = {cname: 0 for cname in class_names}
    
    print("Saving images to disk...")
    for idx, sample in enumerate(train_split):
        img = sample['image']
        label_idx = sample['label']
        class_name = class_names[label_idx]
        
        # Create class directory if not exists
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Determine filename
        counter = class_counters[class_name]
        filename = f"{class_name}_{counter:02d}.png"
        filepath = os.path.join(class_dir, filename)
        
        # Save image
        img.save(filepath, "PNG")
        class_counters[class_name] += 1
        
        if (idx + 1) % 300 == 0:
            print(f"Saved {idx + 1}/{len(train_split)} images...")
            
    print("Extraction complete! Dataset structure:")
    for cname in class_names:
        cpath = os.path.join(output_dir, cname)
        n_files = len(os.listdir(cpath)) if os.path.exists(cpath) else 0
        print(f" - {cname}: {n_files} images")

if __name__ == "__main__":
    # Resolve relative path based on workspace root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_dir = os.path.join(project_root, "data", "raw", "uc_merced")
    download_and_extract_ucm(target_dir)

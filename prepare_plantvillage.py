import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

def main():
    # 1. Config paths
    dataset_dir = "/home/namvh2/TLU_TVQH/SGP_Citrus/Paper_3_SGP-Citrus/Dataset/Public/PlantVillage/PlantVillage"
    output_dir = "/home/namvh2/TLU_TVQH/SGP_Citrus/Paper_3_SGP-Citrus/Dataset/Public/PlantVillage_Features"
    pt_dir = os.path.join(output_dir, "pt_files")
    split_dir = "/home/namvh2/TLU_TVQH/SGP_Citrus/Paper_3_SGP-Citrus/Dataset/Public/PlantVillage_splits"
    
    os.makedirs(pt_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # Classes mapping
    class_mapping = {
        "Tomato_healthy": 0,
        "Tomato_Early_blight": 1,
        "Tomato_Late_blight": 2,
        "Tomato_Septoria_leaf_spot": 3,
        "Tomato__Tomato_YellowLeaf__Curl_Virus": 4
    }

    # 2. Find all images and sample 300 per class
    import random
    random.seed(2025)
    
    image_list = []
    labels = []
    slide_ids = []
    case_ids = []

    print("Scanning dataset folders...")
    for class_name, label_idx in class_mapping.items():
        class_folder = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"Warning: folder {class_folder} not found!")
            continue
        
        # Find all images for this class
        class_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            class_images.extend(glob.glob(os.path.join(class_folder, ext)))
            
        total_found = len(class_images)
        # Randomly sample 300 images
        if len(class_images) > 300:
            class_images = random.sample(class_images, 300)
            
        print(f"Class {class_name}: selected {len(class_images)} images (sampled out of {total_found} total).")
        
        for img_path in class_images:
            image_list.append(img_path)
            labels.append(label_idx)
            # slide_id should be unique and not contain dot except for file extension (removed)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            slide_id = f"{class_name}_{base_name}"
            slide_ids.append(slide_id)
            case_ids.append(slide_id)

    print(f"Total dataset size: {len(image_list)} images across 5 classes.")


    # 3. Setup backbone feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for feature extraction")
    
    # Load ResNet-50
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    resnet = models.resnet50(weights=weights)
    # Remove final classification layer, keep feature map pooling output (2048 dims)
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)
    backbone = backbone.to(device)
    backbone.eval()

    # Image transform
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Extract features
    print("Extracting features (saving to .pt files)...")
    records = []
    with torch.no_grad():
        for i, img_path in enumerate(tqdm(image_list)):
            slide_id = slide_ids[i]
            label = labels[i]
            case_id = case_ids[i]
            
            # Check if already processed
            pt_path = os.path.join(pt_dir, f"{slide_id}.pt")
            if os.path.exists(pt_path):
                records.append({
                    "case_id": case_id,
                    "slide_id": slide_id,
                    "label": label
                })
                continue

            # Extract feature
            try:
                img = Image.open(img_path).convert('RGB')
                tensor = preprocess(img).unsqueeze(0).to(device) # Shape [1, 3, 224, 224]
                features = backbone(tensor) # Shape [1, 2048, 1, 1]
                features = features.squeeze(-1).squeeze(-1) # Shape [1, 2048]
                
                # Save feature tensor
                torch.save(features.cpu(), pt_path)
                
                records.append({
                    "case_id": case_id,
                    "slide_id": slide_id,
                    "label": label
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


    # Save labels.csv
    df_labels = pd.DataFrame(records)
    labels_csv_path = os.path.join(output_dir, "labels.csv")
    df_labels.to_csv(labels_csv_path, index=False)
    print(f"Saved labels.csv to {labels_csv_path}")

    # 5. Stratified 5-Fold Splits
    print("Generating 5-Fold Splits...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
    
    X = np.array(df_labels["slide_id"])
    y = np.array(df_labels["label"])

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y)):
        # Split train_val into train (90%) and val (10%)
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        
        # We want validation set to be about 10% of total data
        # which is 12.5% of train_val data (0.1 / 0.8 = 0.125)
        skf_val = StratifiedKFold(n_splits=8, shuffle=True, random_state=2025)
        train_idx_sub, val_idx_sub = next(skf_val.split(X_train_val, y_train_val))
        
        train_slides = X_train_val[train_idx_sub]
        val_slides = X_train_val[val_idx_sub]
        test_slides = X[test_idx]
        
        # Save split
        df_train = pd.DataFrame({'train': train_slides})
        df_val = pd.DataFrame({'val': val_slides})
        df_test = pd.DataFrame({'test': test_slides})
        df_split = pd.concat([df_train, df_val, df_test], axis=1)
        
        split_file_path = os.path.join(split_dir, f"splits_{fold_idx}.csv")
        df_split.to_csv(split_file_path, index=False)
        print(f"Saved fold {fold_idx} split to {split_file_path}")

if __name__ == "__main__":
    main()

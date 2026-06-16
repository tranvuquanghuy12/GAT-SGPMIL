import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Cấu hình đường dẫn
DATA_DIR = Path("./data/citrus_custom")
IMAGE_RAW_DIR = DATA_DIR / "images_raw"
FEATURES_DIR = DATA_DIR / "features" / "pt_files"
LABEL_FILE = DATA_DIR / "labels.csv"
SPLIT_DIR = DATA_DIR / "splits"

# Cấu hình tiền xử lý
PATCH_SIZE = 224
OVERLAP = 0.5  # 50% overlap
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nhãn tương ứng
CLASS_MAP = {
    "healthy_citrus": 0,
    "citrus_hlb": 1,
    "citrus_canker": 2,
    "citrus_anthracnose": 3,
    "citrus_nutrition_deficiency": 4
}

class FeatureExtractor:
    def __init__(self):
        print(f"📦 Đang khởi tạo ResNet50 Extractor trên thiết bị: {DEVICE}...")
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_patches(self, img: Image.Image) -> list:
        """Cắt ảnh thành các patch kích thước 224x224."""
        w, h = img.size
        patches = []
        stride = int(PATCH_SIZE * (1 - OVERLAP))
        
        for y in range(0, h - PATCH_SIZE + 1, stride):
            for x in range(0, w - PATCH_SIZE + 1, stride):
                patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                patches.append(patch)
                
        if not patches:
            patches.append(img.resize((PATCH_SIZE, PATCH_SIZE)))
            
        return patches

    @torch.no_grad()
    def process_image(self, img_path: Path) -> torch.Tensor:
        """Tiền xử lý, cắt patch và chạy qua mạng ResNet50."""
        img = Image.open(img_path).convert('RGB')
        patches = self.extract_patches(img)
        
        patch_tensors = []
        for patch in patches:
            tensor = self.transform(patch)
            patch_tensors.append(tensor)
            
        batch = torch.stack(patch_tensors).to(DEVICE)
        features = self.model(batch)
        features = features.squeeze(-1).squeeze(-1) # [N_patches, 2048]
        return features.cpu()

def create_dataset_splits(df: pd.DataFrame):
    """Phân chia dữ liệu train/val/test theo tỷ lệ 70/15/15."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Đảm bảo phân chia đồng đều tỷ lệ các lớp (stratified split)
    train_list, val_list, test_list = [], [], []
    
    for label in df['label'].unique():
        df_class = df[df['label'] == label].sample(frac=1, random_state=42)
        n = len(df_class)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        
        train_list.append(df_class.iloc[:n_train])
        val_list.append(df_class.iloc[n_train:n_train+n_val])
        test_list.append(df_class.iloc[n_train+n_val:])
        
    df_train = pd.concat(train_list)
    df_val = pd.concat(val_list)
    df_test = pd.concat(test_list)
    
    split_data = {
        "train": df_train['slide_id'].reset_index(drop=True),
        "val": df_val['slide_id'].reset_index(drop=True),
        "test": df_test['slide_id'].reset_index(drop=True),
    }
    
    df_split = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in split_data.items()]))
    split_file = SPLIT_DIR / "custom_split_0.csv"
    df_split.to_csv(split_file, index=False)
    print(f"✨ Đã tạo file phân chia splits tại: {split_file}")

def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Đọc danh sách ảnh
    image_paths = []
    for suffix in ['.jpg', '.jpeg', '.png']:
        image_paths.extend(list(IMAGE_RAW_DIR.glob(f"**/*{suffix}")))
        
    if not image_paths:
        print("❌ Không tìm thấy ảnh nào trong data/citrus_custom/images_raw/. Hãy chạy file cào ảnh scrape_images.py trước!")
        return

    extractor = FeatureExtractor()
    metadata = []
    success_count = 0
    
    print(f"⚙️ Bắt đầu trích xuất đặc trưng cho {len(image_paths)} ảnh...")
    
    for img_path in image_paths:
        class_name = img_path.parent.name
        class_idx = CLASS_MAP.get(class_name)
        
        if class_idx is None:
            continue
            
        slide_id = img_path.stem
        output_pt_path = FEATURES_DIR / f"{slide_id}.pt"
        
        try:
            # Kiểm tra ảnh có bị lỗi không trước khi nạp
            with Image.open(img_path) as test_img:
                test_img.verify()
                
            # Trích xuất đặc trưng
            features = extractor.process_image(img_path)
            
            # Lưu đặc trưng
            torch.save(features, output_pt_path)
            
            metadata.append({
                "slide_id": slide_id,
                "label": class_idx
            })
            success_count += 1
            print(f"  [Processed {success_count}/{len(image_paths)}] Trích xuất thành công: {slide_id}.pt (Patches: {features.shape[0]})", end='\r')
        except Exception:
            # Bỏ qua âm thầm các ảnh bị lỗi/hỏng trong quá trình tải về
            continue
            
    # Ghi nhận labels.csv
    df = pd.DataFrame(metadata)
    df.to_csv(LABEL_FILE, index=False)
    print(f"\n✨ Đã lưu danh sách nhãn tại: {LABEL_FILE}")
    
    # Tạo phân chia splits
    create_dataset_splits(df)
    print("🎉 Hoàn tất toàn bộ quy trình tiền xử lý và trích xuất đặc trưng!")

if __name__ == "__main__":
    main()

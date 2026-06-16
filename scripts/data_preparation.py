import os
import urllib.request
import json
import re
import socket
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Thiết lập timeout để tránh treo khi tải ảnh
socket.setdefaulttimeout(15)

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DATA_DIR = Path("./data/citrus_custom")
IMAGE_RAW_DIR = DATA_DIR / "images_raw"
FEATURES_DIR = DATA_DIR / "features" / "pt_files"
LABEL_FILE = DATA_DIR / "labels.csv"
SPLIT_DIR = DATA_DIR / "splits"

# Định nghĩa 5 nhãn và từ khóa tìm kiếm trên Bing để cào ảnh
KEYWORDS = {
    0: {"name": "healthy_citrus", "query": "healthy citrus leaf"},
    1: {"name": "citrus_hlb", "query": "citrus greening disease huanglongbing leaf"},
    2: {"name": "citrus_canker", "query": "citrus canker leaf"},
    3: {"name": "citrus_anthracnose", "query": "citrus anthracnose leaf disease"},
    4: {"name": "citrus_nutrition_deficiency", "query": "citrus leaf nutrient deficiency yellowing"}
}

NUM_IMAGES_PER_CLASS = 200 # Số lượng ảnh muốn cào cho mỗi lớp
PATCH_SIZE = 224
OVERLAP = 0.5  # 50% overlap giữa các patch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PHẦN 1: BỘ CÀO ẢNH TỪ BING (IMAGE SCRAPER)
# ==========================================
def fetch_image_urls(query: str, limit: int = 200) -> list:
    """Tìm kiếm link ảnh trên Bing Images không cần API Key."""
    print(f"🔍 Đang tìm kiếm ảnh cho từ khóa: '{query}'...")
    urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Duyệt qua các trang kết quả của Bing
    for offset in range(0, limit + 50, 50):
        url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&first={offset}&count=50"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                html = response.read().decode('utf-8')
                # Trích xuất link ảnh gốc từ JSON trong HTML của Bing
                m_matches = re.findall(r'm="({.*?})"', html)
                for m in m_matches:
                    try:
                        m_clean = m.replace('&quot;', '"')
                        data = json.loads(m_clean)
                        if 'murl' in data:
                            urls.append(data['murl'])
                    except Exception:
                        continue
        except Exception as e:
            print(f"⚠️ Lỗi khi kết nối Bing (offset {offset}): {e}")
            break
            
        if len(urls) >= limit:
            break
            
    return list(set(urls))[:limit]

def download_images():
    """Tải ảnh từ các URL đã cào về thư mục tương ứng."""
    IMAGE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for class_idx, info in KEYWORDS.items():
        class_name = info["name"]
        query = info["query"]
        class_dir = IMAGE_RAW_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Lấy danh sách link ảnh
        urls = fetch_image_urls(query, limit=NUM_IMAGES_PER_CLASS)
        print(f"Found {len(urls)} URLs for class: {class_name}")
        
        downloaded = 0
        for i, url in enumerate(urls):
            file_extension = ".jpg"  # Mặc định là jpg
            if ".png" in url.lower():
                file_extension = ".png"
                
            dest_path = class_dir / f"{class_name}_{i:03d}{file_extension}"
            
            # Bỏ qua nếu file đã tồn tại
            if dest_path.exists():
                downloaded += 1
                continue
                
            try:
                # Thiết lập User-Agent giả lập trình duyệt để tránh bị chặn khi tải ảnh
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                downloaded += 1
                print(f"  [+] Tải thành công {downloaded}/{NUM_IMAGES_PER_CLASS}: {dest_path.name}", end='\r')
            except Exception:
                # Bỏ qua các link lỗi, không in chi tiết để tránh làm rối màn hình
                continue
        print(f"\n✅ Hoàn thành lớp '{class_name}': Đã tải {downloaded} ảnh.")

# ==========================================
# PHẦN 2: CẮT PATCH VÀ TRÍCH XUẤT ĐẶC TRƯNG
# ==========================================
class FeatureExtractor:
    def __init__(self, model_name="resnet50"):
        print(f"📦 Đang khởi tạo bộ trích xuất đặc trưng ({model_name})...")
        if model_name == "resnet50":
            # Tải ResNet50 pre-trained trên ImageNet
            weights = models.ResNet50_Weights.DEFAULT
            base_model = models.resnet50(weights=weights)
            # Loại bỏ lớp phân loại FC cuối cùng để lấy đặc trưng 2048 chiều
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        else:
            raise NotImplementedError("Hiện tại script chỉ hỗ trợ 'resnet50'")
            
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        # Chuẩn hóa ảnh theo chuẩn của ImageNet
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_patches(self, img: Image.Image) -> list:
        """Cắt ảnh thành các patch nhỏ kích thước 224x224 có overlap."""
        w, h = img.size
        patches = []
        stride = int(PATCH_SIZE * (1 - OVERLAP))
        
        for y in range(0, h - PATCH_SIZE + 1, stride):
            for x in range(0, w - PATCH_SIZE + 1, stride):
                patch = img.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
                patches.append(patch)
                
        # Nếu ảnh quá nhỏ không cắt được patch nào, tự động resize và lấy trung tâm
        if not patches:
            patches.append(img.resize((PATCH_SIZE, PATCH_SIZE)))
            
        return patches

    @torch.no_grad()
    def process_image(self, img_path: Path) -> torch.Tensor:
        """Cắt các patch của 1 ảnh, trích xuất vector đặc trưng và xếp chồng thành 1 Tensor."""
        img = Image.open(img_path).convert('RGB')
        patches = self.extract_patches(img)
        
        patch_tensors = []
        for patch in patches:
            tensor = self.transform(patch)
            patch_tensors.append(tensor)
            
        # [N_patches, 3, 224, 224]
        batch = torch.stack(patch_tensors).to(DEVICE)
        
        # Trích xuất đặc trưng
        features = self.model(batch) # [N_patches, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1) # [N_patches, 2048]
        return features.cpu()

def process_all_images():
    """Duyệt qua tất cả ảnh đã tải, trích xuất đặc trưng và tạo file nhãn labels.csv."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor("resnet50")
    
    metadata = []
    
    # Lấy toàn bộ đường dẫn ảnh
    image_paths = list(IMAGE_RAW_DIR.glob("**/*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"⚙️ Bắt đầu tiền xử lý và trích xuất đặc trưng cho {len(image_paths)} ảnh...")
    
    success_count = 0
    for idx, img_path in enumerate(image_paths):
        # Xác định nhãn lớp dựa vào tên thư mục cha
        class_name = img_path.parent.name
        class_idx = next((k for k, v in KEYWORDS.items() if v["name"] == class_name), None)
        
        if class_idx is None:
            continue
            
        slide_id = img_path.stem
        output_pt_path = FEATURES_DIR / f"{slide_id}.pt"
        
        try:
            # Trích xuất đặc trưng
            features = extractor.process_image(img_path)
            
            # Lưu file tensor [N_patches, Feature_dim]
            torch.save(features, output_pt_path)
            
            # Ghi nhận nhãn
            metadata.append({
                "slide_id": slide_id,
                "label": class_idx
            })
            success_count += 1
            print(f"  [Processed {success_count}/{len(image_paths)}] Trích xuất thành công {slide_id}.pt (Patches: {features.shape[0]})", end='\r')
        except Exception as e:
            print(f"\n⚠️ Lỗi khi xử lý file {img_path.name}: {e}")
            continue
            
    # Tạo file labels.csv
    df = pd.DataFrame(metadata)
    df.to_csv(LABEL_FILE, index=False)
    print(f"\n✨ Đã tạo thành công file nhãn tại: {LABEL_FILE}")
    
    # Tạo phân chia thử nghiệm Train/Val/Test đơn giản (splits)
    create_dataset_splits(df)

def create_dataset_splits(df: pd.DataFrame):
    """Phân chia dữ liệu thành Train (70%), Val (15%), Test (15%) và lưu file split."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Trộn ngẫu nhiên dữ liệu
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_shuffled)
    
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    df_shuffled.loc[:n_train, 'split'] = 'train'
    df_shuffled.loc[n_train:n_train+n_val, 'split'] = 'val'
    df_shuffled.loc[n_train+n_val:, 'split'] = 'test'
    
    # Lưu định dạng split giống như SGPMIL yêu cầu
    # Thường SGPMIL yêu cầu các cột: train, val, test chứa slide_id
    split_data = {
        "train": df_shuffled[df_shuffled['split'] == 'train']['slide_id'].reset_index(drop=True),
        "val": df_shuffled[df_shuffled['split'] == 'val']['slide_id'].reset_index(drop=True),
        "test": df_shuffled[df_shuffled['split'] == 'test']['slide_id'].reset_index(drop=True),
    }
    
    df_split = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in split_data.items() ]))
    split_file = SPLIT_DIR / "custom_split_0.csv"
    df_split.to_csv(split_file, index=False)
    print(f"✨ Đã tạo file phân chia splits tại: {split_file}")

# ==========================================
# LUỒNG CHẠY CHÍNH
# ==========================================
if __name__ == "__main__":
    print("🚀 Bắt đầu quy trình chuẩn bị dữ liệu...")
    # Bước 1: Tải ảnh
    download_images()
    # Bước 2: Cắt patch và trích xuất đặc trưng
    process_all_images()
    print("🎉 Hoàn tất toàn bộ quy trình! Dữ liệu đã sẵn sàng để huấn luyện.")

import os
import urllib.request
import urllib.parse
import json
import re
import socket
import sys
from pathlib import Path

# Thiết lập timeout để tránh bị treo
socket.setdefaulttimeout(15)

# Cấu hình đường dẫn
DATA_DIR = Path("./data/citrus_custom")
IMAGE_RAW_DIR = DATA_DIR / "images_raw"

# Định nghĩa các lớp và các từ khóa phụ (Search Queries) nhằm cào sâu hơn, đạt đủ 200 ảnh sạch
KEYWORDS = {
    0: {
        "name": "healthy_citrus", 
        "queries": ["healthy citrus leaf close up", "healthy orange leaf", "healthy lemon tree leaf", "clean citrus foliage"]
    },
    1: {
        "name": "citrus_hlb", 
        "queries": ["citrus greening disease leaf", "huanglongbing orange leaf", "citrus hlb symptoms leaves", "liberibacter asiaticus citrus leaf"]
    },
    2: {
        "name": "citrus_canker", 
        "queries": ["citrus canker leaf symptoms", "xanthomonas citri orange leaf", "citrus canker disease spots", "lemon leaf canker spot"]
    },
    3: {
        "name": "citrus_anthracnose", 
        "queries": ["citrus anthracnose leaf", "colletotrichum citrus leaf disease", "orange leaf anthracnose decay", "citrus withertip leaf"]
    },
    4: {
        "name": "citrus_nutrition_deficiency", 
        "queries": ["citrus leaf zinc deficiency", "citrus leaf nitrogen deficiency yellowing", "citrus magnesium deficiency leaf", "orange tree nutrient deficiency leaves"]
    }
}

NUM_IMAGES_PER_CLASS = 200

def fetch_image_urls(queries: list, limit: int = 300) -> list:
    """Tìm kiếm link ảnh trên Bing qua nhiều từ khóa phụ để gom đủ số lượng."""
    urls = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Duyệt qua từng truy vấn phụ để đa dạng hóa nguồn ảnh
    for query in queries:
        if len(urls) >= limit:
            break
        print(f"  🔍 Đang quét từ khóa: '{query}'...")
        
        # Cào sâu qua nhiều trang (offset 0 -> 150)
        for offset in range(0, 150, 50):
            url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}&first={offset}&count=50"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    html = response.read().decode('utf-8')
                    m_matches = re.findall(r'm="({.*?})"', html)
                    for m in m_matches:
                        try:
                            m_clean = m.replace('&quot;', '"')
                            data = json.loads(m_clean)
                            if 'murl' in data and data['murl'] not in urls:
                                urls.append(data['murl'])
                        except Exception:
                            continue
            except Exception as e:
                # Bỏ qua lỗi kết nối tạm thời của Bing
                continue
                
            if len(urls) >= limit:
                break
                
    return list(set(urls))[:limit]

def download_images():
    """Tải ảnh và đảm bảo đạt tối thiểu 200 ảnh thực tế cho mỗi lớp."""
    IMAGE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    for class_idx, info in KEYWORDS.items():
        class_name = info["name"]
        queries = info["queries"]
        class_dir = IMAGE_RAW_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n========================================\n🚀 BẮT ĐẦU CÀO LỚP: {class_name}\n========================================")
        
        # Thu thập tối đa 350 URLs đề phòng link hỏng/chết
        urls = fetch_image_urls(queries, limit=350)
        print(f"Tổng số URL ảnh tìm thấy cho {class_name}: {len(urls)}")
        
        # Đếm số ảnh hợp lệ đã có sẵn
        downloaded = len([p for p in class_dir.glob("*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        print(f"Số ảnh đã có sẵn trong thư mục: {downloaded}")
        
        if downloaded >= NUM_IMAGES_PER_CLASS:
            print(f"✅ Đã có đủ {downloaded} ảnh cho lớp {class_name}. Bỏ qua.")
            continue
            
        url_idx = 0
        while downloaded < NUM_IMAGES_PER_CLASS and url_idx < len(urls):
            url = urls[url_idx]
            url_idx += 1
            
            # Xác định định dạng
            file_extension = ".jpg"
            if ".png" in url.lower():
                file_extension = ".png"
                
            dest_path = class_dir / f"{class_name}_{downloaded:03d}{file_extension}"
            
            try:
                # Request tải ảnh giả lập User-Agent
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    data = response.read()
                    
                    # Kiểm tra file tải về có dung lượng tối thiểu (> 5KB) để tránh ảnh lỗi, ảnh trống
                    if len(data) < 5120: 
                        continue
                        
                    with open(dest_path, 'wb') as f:
                        f.write(data)
                
                downloaded += 1
                sys.stdout.write(f"\r  [+] Tiến trình tải: {downloaded}/{NUM_IMAGES_PER_CLASS} ảnh (File: {dest_path.name})")
                sys.stdout.flush()
            except Exception:
                # Bỏ qua lỗi tải
                continue
                
        print(f"\n✨ Kết quả: Đã tải xong lớp '{class_name}'. Tổng số ảnh: {downloaded}")
        if downloaded < NUM_IMAGES_PER_CLASS:
            print(f"⚠️ Cảnh báo: Chỉ tải được {downloaded}/{NUM_IMAGES_PER_CLASS} ảnh cho lớp {class_name}. Bạn nên kiểm tra lại kết nối mạng.")

if __name__ == "__main__":
    print("🌐 Bắt đầu chương trình cào ảnh nông nghiệp tự động...")
    download_images()
    print("\n🎉 Hoàn thành cào dữ liệu ảnh! Hãy kiểm tra chất lượng ảnh thô trong thư mục: data/citrus_custom/images_raw/")

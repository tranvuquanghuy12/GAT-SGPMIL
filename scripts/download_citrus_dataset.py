#!/usr/bin/env python3
"""
download_citrus_dataset.py
==========================
Tự động cào ảnh lá cam/quýt bị vàng lá gân xanh (HLB) và lá khoẻ mạnh từ Google Images.
Sử dụng thư viện icrawler đa luồng cực nhanh để tạo bộ dữ liệu MIL cho bài báo Citrus-GAT-SGPMIL.

Cài đặt thư viện bổ trợ:
    pip install icrawler
"""

import os
import sys
from icrawler.builtin import GoogleImageCrawler

def setup_directories(base_dir="data/citrus_dataset"):
    """Tạo cấu trúc thư mục lưu trữ ảnh"""
    pos_dir = os.path.join(base_dir, "positive_hlb")
    neg_dir = os.path.join(base_dir, "negative_healthy")
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)
    return pos_dir, neg_dir

def download_images():
    print("=" * 60)
    print(" 🔬 BẮT ĐẦU CÀO BỘ DỮ LIỆU LÁ CAM QUÝT (HLB vs HEALTHY)")
    print("=" * 60)

    # 1. Định cấu hình thư mục lưu
    pos_dir, neg_dir = setup_directories()

    # 2. Định nghĩa từ khóa học thuật chuẩn xác
    # Từ khoá positive (Lá bệnh vàng lá gân xanh - HLB)
    hlb_keywords = [
        "citrus greening disease leaves",
        "huanglongbing citrus leaves symptoms",
        "citrus HLB leaf mottle",
        "yellow dragon disease citrus leaves",
        "bệnh vàng lá gân xanh cam quýt"
    ]

    # Từ khoá negative (Lá khoẻ mạnh)
    healthy_keywords = [
        "healthy citrus leaves",
        "healthy orange tree leaf",
        "healthy lemon tree leaves",
        "lá cam khỏe mạnh",
        "lá bưởi xanh tươi"
    ]

    # Số lượng ảnh cần tải cho mỗi từ khóa (tổng số ảnh kỳ vọng ~400-500 ảnh mỗi lớp)
    max_images_per_keyword = 120

    # 3. Tiến hành cào lớp Bệnh (Positive)
    print("\n[+] ĐANG TẢI ẢNH LÁ BỆNH (POSITIVE - HLB)...")
    for idx, keyword in enumerate(hlb_keywords):
        print(f"  -> Đang cào từ khóa {idx+1}/{len(hlb_keywords)}: '{keyword}'")
        # Sử dụng Crawler của Google
        google_crawler = GoogleImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={'root_dir': pos_dir}
        )
        # Bộ lọc lọc kích thước ảnh (tránh ảnh quá nhỏ hoặc icon)
        google_crawler.crawl(
            keyword=keyword,
            filters=dict(size='medium'),
            max_num=max_images_per_keyword,
            file_idx_offset='auto'
        )

    # 4. Tiến hành cào lớp Khỏe mạnh (Negative)
    print("\n[+] ĐANG TẢI ẢNH LÁ KHỎE MẠNH (NEGATIVE - HEALTHY)...")
    for idx, keyword in enumerate(healthy_keywords):
        print(f"  -> Đang cào từ khóa {idx+1}/{len(healthy_keywords)}: '{keyword}'")
        google_crawler = GoogleImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={'root_dir': neg_dir}
        )
        google_crawler.crawl(
            keyword=keyword,
            filters=dict(size='medium'),
            max_num=max_images_per_keyword,
            file_idx_offset='auto'
        )

    # 5. Tổng kết
    pos_count = len([f for f in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, f))])
    neg_count = len([f for f in os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, f))])

    print("\n" + "=" * 60)
    print(" 🎉 HOÀN THÀNH THU THẬP BỘ DỮ LIỆU")
    print(f"   - Thư mục lưu: data/citrus_dataset/")
    print(f"   - Tổng số ảnh Bệnh (Positive): {pos_count} ảnh")
    print(f"   - Tổng số ảnh Khỏe mạnh (Negative): {neg_count} ảnh")
    print("=" * 60)

if __name__ == "__main__":
    download_images()

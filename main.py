# main.py
"""
Chạy RAG chatbot từ command line
"""
import sys
import os

# Thêm src vào đường dẫn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import setup_api_key
from src.rag_utils import SimpleRAG
from src.doc_utils import load_documents_from_folder
from src.web_scraper import scrape_web

def main():
    print("="*60)
    print("🤖 RAG Chatbot - Hỏi đáp dựa trên tài liệu")
    print("="*60)
    
    # Thiết lập API key
    api_key = setup_api_key()
    
    # Tạo RAG
    rag = SimpleRAG(api_key)
    
    # ===== PHẦN ĐỌC TÀI LIỆU MỚI =====
    data_folder = "data"  # Thư mục chứa tài liệu

    # Web scraping
    scrape_web()
    
    # File scraping (bao gồm cả file tổng hợp web scraping)
    print(f"\n📂 Đang đọc tài liệu từ thư mục: {data_folder}/")
    tai_lieu = load_documents_from_folder(data_folder)
    
    if not tai_lieu:
        print("⚠️  Không tìm thấy tài liệu nào trong thư mục data/")
        print("   Hỗ trợ các định dạng: .txt, .pdf, .docx")
        return
    else:
        print(f"\n✅ Đã đọc {len(tai_lieu)} tài liệu")
        
    
    # Nạp tài liệu
    rag.load_documents(tai_lieu)
    
    # Vòng lặp hỏi đáp
    print("\n💬 Nhập câu hỏi (gõ 'thoat' để kết thúc, 'stats' để xem thông tin):\n")
    
    while True:
        cau_hoi = input("Bạn: ").strip()
        
        if cau_hoi.lower() == 'thoat':
            print("👋 Tạm biệt!")
            break
        elif cau_hoi.lower() == 'stats':
            stats = rag.get_stats()
            print("\n📊 Thông tin RAG:")
            for k, v in stats.items():
                print(f"  • {k}: {v}")
            continue
        elif not cau_hoi:
            continue
        
        # Trả lời
        cau_tra_loi = rag.ask(cau_hoi)
        print(f"\n🤖 Bot: {cau_tra_loi}\n")

if __name__ == "__main__":
    main()
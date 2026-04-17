import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from datetime import datetime
from .constants import POLICIES, PRODUCTS

class DataScraper:
    """Lớp thu thập dữ liệu từ các nguồn"""
    
    def __init__(self, delay=1):
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.policies_data = []      # ← Đổi tên: lưu dữ liệu chính sách
        self.products_data = []       # ← Đổi tên: lưu dữ liệu mặt hàng
    
    def fetch_page(self, url):
        """Tải nội dung trang web"""
        try:
            print(f"🌐 Đang tải: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            return None
    
    def extract_text(self, html):
        """Trích xuất văn bản từ HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    # ========== HÀM CHO CHÍNH SÁCH ==========
    def scrape_policy(self, url, source_name):
        """
        Thu thập dữ liệu chính sách đổi trả hàng
        
        Args:
            url (str): URL trang chính sách
            source_name (str): Tên doanh nghiệp/nguồn
        """
        html = self.fetch_page(url)
        if not html:
            return None
        
        text = self.extract_text(html)
        
        record = {
            'type': 'policy',           # ← Thêm loại dữ liệu
            'source': source_name,
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'content': text,
            'word_count': len(text.split())
        }
        
        self.policies_data.append(record)  # ← Lưu vào mảng riêng
        print(f"   📜 Đã thu thập chính sách từ {source_name}: {record['word_count']} từ")
        
        time.sleep(self.delay)
        return record
    
    # ========== HÀM CHO DANH SÁCH MẶT HÀNG ==========
    def scrape_product_list(self, url, category_name):
        """
        Thu thập danh sách mặt hàng từ trang web
        
        Args:
            url (str): URL trang danh sách sản phẩm
            category_name (str): Tên danh mục (điện tử, thời trang...)
        """
        html = self.fetch_page(url)
        if not html:
            return None
        
        text = self.extract_text(html)
        
        record = {
            'type': 'product_list',     # ← Thêm loại dữ liệu
            'category': category_name,
            'url': url,
            'scraped_at': datetime.now().isoformat(),
            'content': text,
            'word_count': len(text.split())
        }
        
        self.products_data.append(record)  # ← Lưu vào mảng riêng
        print(f"   🛒 Đã thu thập danh sách mặt hàng {category_name}: {record['word_count']} từ")
        
        time.sleep(self.delay)
        return record
    
    # ========== HÀM LƯU DỮ LIỆU ==========
    def save_all_to_txt(self, output_dir='data/raw'):
        """
        Lưu tất cả dữ liệu vào file .txt
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu dữ liệu chính sách
        for record in self.policies_data:
            safe_name = record['source'].replace(' ', '_').replace('/', '_')
            filename = f"{output_dir}/policy_{safe_name}_{record['scraped_at'][:10]}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"[LOẠI DỮ LIỆU: CHÍNH SÁCH]\n")
                f.write(f"Nguồn: {record['source']}\n")
                f.write(f"URL: {record['url']}\n")
                f.write(f"Ngày: {record['scraped_at']}\n")
                f.write("="*60 + "\n\n")
                f.write(record['content'])
            
            print(f"💾 Đã lưu chính sách: {filename}")
        
        # Lưu dữ liệu danh sách mặt hàng
        for record in self.products_data:
            safe_name = record['category'].replace(' ', '_').replace('/', '_')
            filename = f"{output_dir}/products_{safe_name}_{record['scraped_at'][:10]}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"[LOẠI DỮ LIỆU: DANH SÁCH MẶT HÀNG]\n")
                f.write(f"Danh mục: {record['category']}\n")
                f.write(f"URL: {record['url']}\n")
                f.write(f"Ngày: {record['scraped_at']}\n")
                f.write("="*60 + "\n\n")
                f.write(record['content'])
            
            print(f"💾 Đã lưu danh sách mặt hàng: {filename}")
    
    def save_combined(self, output_file='data/combined_data.txt'):
        """
        Lưu tất cả dữ liệu vào một file duy nhất
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Ghi phần chính sách
            f.write("\n\n" + "█"*60 + "\n")
            f.write("█  PHẦN 1: CHÍNH SÁCH ĐỔI TRẢ HÀNG\n")
            f.write("█"*60 + "\n\n")
            
            for record in self.policies_data:
                f.write(f"\n{'='*60}\n")
                f.write(f"NGUỒN: {record['source']}\n")
                f.write(f"URL: {record['url']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(record['content'])
                f.write("\n\n")
            
            # Ghi phần danh sách mặt hàng
            f.write("\n\n" + "█"*60 + "\n")
            f.write("█  PHẦN 2: DANH SÁCH MẶT HÀNG\n")
            f.write("█"*60 + "\n\n")
            
            for record in self.products_data:
                f.write(f"\n{'='*60}\n")
                f.write(f"DANH MỤC: {record['category']}\n")
                f.write(f"URL: {record['url']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(record['content'])
                f.write("\n\n")
        
        print(f"📚 Đã lưu file tổng hợp: {output_file}")

    def scrape_web(self):
        # Xóa file cũ
        combined_file = 'data/combined_data.txt'
        raw_folder = 'data/raw'
        
        if os.path.exists(combined_file):
            os.remove(combined_file)
            print(f"🗑️  Đã xóa file cũ: {combined_file}")
        
        if os.path.exists(raw_folder):
            for file in os.listdir(raw_folder):
                file_path = os.path.join(raw_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"🗑️  Đã xóa các file cũ trong {raw_folder}/")
        scraper = DataScraper(delay=5)
        
        # ===== 1. THU THẬP CHÍNH SÁCH =====
        print("\n" + "="*60)
        print("📜 THU THẬP CHÍNH SÁCH ĐỔI TRẢ HÀNG")
        print("="*60)
        
        for url, name in POLICIES:
            scraper.scrape_policy(url, name)
        
        # ===== 2. THU THẬP DANH SÁCH MẶT HÀNG =====
        print("\n" + "="*60)
        print("🛒 THU THẬP DANH SÁCH MẶT HÀNG")
        print("="*60)

        for url, category in PRODUCTS:
            scraper.scrape_product_list(url, category)
        
        # ===== 3. LƯU DỮ LIỆU =====
        print("\n" + "="*60)
        print("💾 LƯU DỮ LIỆU")
        print("="*60)
        
        scraper.save_all_to_txt()
        scraper.save_combined()
        
        # ===== 4. THỐNG KÊ =====
        print("\n" + "="*60)
        print("📊 THỐNG KÊ")
        print("="*60)
        print(f"   • Chính sách: {len(scraper.policies_data)} nguồn")
        print(f"   • Danh sách mặt hàng: {len(scraper.products_data)} danh mục")
        print(f"   • Tổng số: {len(scraper.policies_data) + len(scraper.products_data)} nguồn")
        
        print("\n✨ HOÀN TẤT!")
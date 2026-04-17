import os
from dotenv import load_dotenv
from pathlib import Path

# Tìm đường dẫn đến file .env (ở thư mục gốc)
BASE_DIR = Path(__file__).parent.parent
ENV_PATH = BASE_DIR / ".env"

# Load file .env
load_dotenv(ENV_PATH)

def setup_api_key():
    """
    Lấy API key từ .env
    Nếu chưa có, yêu cầu nhập và lưu vào .env
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("🔑 Chưa có API key trong file .env")
        print(f"   File .env ở: {ENV_PATH}")
        api_key = input("Nhập Google API key của bạn: ")
        
        # Lưu vào file .env
        with open(ENV_PATH, "a") as f:
            f.write(f"\nGOOGLE_API_KEY={api_key}")
        
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ Đã lưu API key vào file .env")
    
    return api_key

def set_up_hf_token():
    """
    Lấy Hugging Face token từ .env
    Nếu chưa có, yêu cầu nhập và lưu vào .env
    """
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("🔑 Chưa có Hugging Face token trong file .env")
        print(f"   File .env ở: {ENV_PATH}")
        hf_token = input("Nhập Hugging Face token của bạn: ")
        
        # Lưu vào file .env
        with open(ENV_PATH, "a") as f:
            f.write(f"\nHF_TOKEN={hf_token}")
        
        os.environ["HF_TOKEN"] = hf_token
        print("✅ Đã lưu Hugging Face token vào file .env")
    
    return hf_token

# Đọc các cấu hình từ .env (hoặc dùng giá trị mặc định)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 50))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.2))
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-3-flash-preview")
K_RETRIEVAL = int(os.environ.get("K_RETRIEVAL", 8))

print("✅ Đã load cấu hình từ .env")
print(f"   • Model: {MODEL_NAME}")
print(f"   • Chunk size: {CHUNK_SIZE}")
print(f"   • Temperature: {TEMPERATURE}")
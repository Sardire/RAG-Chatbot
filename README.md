1. Chuẩn bị môi trường (Environment Preparation)
   Để dự án hoạt động ổn định, bạn nên sử dụng Python phiên bản 3.10 trở lên. Hãy thực hiện các bước sau trong Terminal hoặc Git Bash:

Tạo môi trường ảo (Virtual Environment): Việc này giúp quản lý các thư viện riêng biệt cho dự án, tránh xung đột hệ thống.

Bash
python -m venv venv
Kích hoạt môi trường ảo:

Trên Windows (Git Bash/CMD):

Bash
source venv/Scripts/activate
Cài đặt thư viện: Sử dụng file requirements.txt để cài đặt toàn bộ các dependency cần thiết (bao gồm LangChain, Google GenAI, ChromaDB, và PyVi).

Bash
pip install -r requirements.txt

2. Cấu hình thông tin (Configuration)
   Hệ thống cần các khóa API để kết nối với mô hình ngôn ngữ và mô hình nhúng (embedding).

Tạo file cấu hình: Tạo một file .env nằm tại thư mục gốc của dự án.

Các tham số cần thiết: Sao chép và điền thông tin của bạn vào file .env

# API Key từ Google AI Studio (Bắt buộc)

GOOGLE_API_KEY=your_google_api_key_here

# Token từ HuggingFace (Cần thiết để tải model embedding)

HF_TOKEN=your_huggingface_token_here

# Cấu hình RAG (Tùy chỉnh nếu cần)

MODEL_NAME=gemini-1.5-flash
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TEMPERATURE=0.2

Cơ chế tự động: Nếu bạn chưa tạo file .env, khi khởi chạy ứng dụng, hàm setup_api_key() sẽ tự động yêu cầu bạn nhập khóa trực tiếp từ Terminal và lưu lại vào file .env cho các lần chạy sau.

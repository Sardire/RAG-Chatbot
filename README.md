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

   Tạo file cấu hình: Tạo một file tên là .env nằm tại thư mục gốc của dự án.

   Các tham số cần thiết: Sao chép và điền thông tin của bạn vào file .env:

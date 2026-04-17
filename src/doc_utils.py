import os
from pathlib import Path

class Converter:
    def read_pdf(self, file_path):
        """Đọc nội dung file PDF"""
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            print(f"❌ Lỗi đọc PDF {file_path}: {e}")
            return ""

    def read_docx(self, file_path):
        """Đọc nội dung file Word (.docx)"""
        try:
            from docx import Document
            doc = Document(file_path)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            print(f"❌ Lỗi đọc Word {file_path}: {e}")
            return ""

    def read_txt(self, file_path):
        """Đọc nội dung file text"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"❌ Lỗi đọc TXT {file_path}: {e}")
            return ""

    def load_documents_from_folder(self, folder_path):
        """
        Đọc tất cả tài liệu trong thư mục
        
        Args:
            folder_path (str): Đường dẫn đến thư mục chứa tài liệu
        
        Returns:
            list: Danh sách nội dung các tài liệu
        """
        
        documents = []
        folder = Path(folder_path)
        
        # Các định dạng hỗ trợ
        supported_extensions = {
            '.txt': self.read_txt,
            '.pdf': self.read_pdf,
            '.docx': self.read_docx
        }
        
        for file_path in folder.iterdir():
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in supported_extensions:
                    print(f"📄 Đang đọc: {file_path.name}")
                    content = supported_extensions[ext](str(file_path))
                    if content:
                        documents.append(content)
                        print(f"   ✅ Đã đọc {len(content)} ký tự")
                else:
                    print(f"⚠️  Bỏ qua: {file_path.name} (không hỗ trợ {ext})")
        
        return documents
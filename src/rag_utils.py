import os
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from pyvi.ViTokenizer import tokenize
from .config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, 
    MODEL_NAME, K_RETRIEVAL
)

# Import từ file .env nếu có

class SimpleRAG:
    
    def __init__(self, api_key=None, hf_token=None):
        """
        Khởi tạo RAG với API key
        
        Args:
            api_key (str): Google API key (nếu None thì lấy từ biến môi trường)
        """
        # Cấu hình API key
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            raise ValueError("❌ Vui lòng cung cấp GOOGLE_API_KEY dưới dạng tham số hoặc biến môi trường.")
        
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        else:
            raise ValueError("❌ Vui lòng cung cấp HF_TOKEN dưới dạng tham số hoặc biến môi trường.")
        
        # Khởi tạo LLM 
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            max_output_tokens=1024,
            google_api_key= os.environ.get("GOOGLE_API_KEY"), # Hoặc thiết lập trong environment
            temperature=0.2
        )
        
        # Dòng mới (sử dụng model embedding mới)
        model_name = "intfloat/multilingual-e5-small"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # 3. Các biến lưu trữ
        self.vectorstore = None   # Kho vector database
        self.qa_chain = None      # Chain xử lý RAG
        self.doc_count = 0        # Số lượng đoạn tài liệu
        
        print("✅ Đã khởi tạo SimpleRAG thành công!")
    
    def load_documents(self, documents):
        """
        Nạp tài liệu vào RAG
        
        Args:
            documents (list): Danh sách các đoạn văn bản hoặc file paths
        
        Returns:
            self: Để có thể gọi liên tiếp
        """
        print(f"📚 Đang xử lý {len(documents)} tài liệu...")
        
        # Bước 1: Chia nhỏ tài liệu thành các đoạn nhỏ
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Tạo các đoạn văn từ tài liệu
        docs = text_splitter.create_documents(documents)
        self.doc_count = len(docs)
        print(f"  → Đã chia thành {self.doc_count} đoạn nhỏ")
        
        # Bước 2: Tạo vector store (biến mỗi đoạn thành vector và lưu lại)
        print("  → Đang tạo vector database...")
        self.vectorstore = Chroma.from_documents(docs, self.embeddings)
        # Bước 3: Tạo chain trả lời (kết hợp tìm kiếm + sinh câu trả lời)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": K_RETRIEVAL}  # Lấy K đoạn liên quan nhất
            )
        )
        
        print(f"✅ Đã nạp xong {self.doc_count} đoạn tài liệu vào RAG!")
        return self
    
    def ask(self, question, show_context=False):
        """
        Đặt câu hỏi cho chatbot
        
        Args:
            question (str): Câu hỏi cần trả lời
            show_context (bool): Có hiển thị đoạn tài liệu được dùng không?
        
        Returns:
            str: Câu trả lời
        """
        if not self.qa_chain:
            return "❌ Chưa có tài liệu. Hãy gọi load_documents() trước."

        # Nếu muốn xem đoạn tài liệu được truy xuất
        if show_context:
            # Lấy các đoạn liên quan
            if self.vectorstore:
                docs = self.vectorstore.similarity_search(question, k=K_RETRIEVAL)
            print("\n📖 Đoạn tài liệu được truy xuất:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc.page_content[:200]}...")
            print("-" * 50)
        
        # Gọi RAG để trả lời
        result = self.qa_chain.invoke({"query": question})
        return result['result']
    
    def ask_batch(self, questions):
        """
        Hỏi nhiều câu hỏi cùng lúc
        
        Args:
            questions (list): Danh sách câu hỏi
        
        Returns:
            dict: Kết quả cho từng câu hỏi
        """
        results = {}
        for q in questions:
            results[q] = self.ask(q)
        return results
    
    def get_stats(self):
        """Lấy thông tin thống kê về RAG"""
        return {
            "doc_count": self.doc_count,
            "chunk_size": CHUNK_SIZE,
            "temperature": TEMPERATURE,
            "k_retrieval": K_RETRIEVAL,
            "model": MODEL_NAME
        }
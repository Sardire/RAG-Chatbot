import os
import pickle
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pyvi.ViTokenizer import tokenize
from .config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE,
    MODEL_NAME, K_RETRIEVAL
)


class SimpleRAG:

    def __init__(self, api_key=None, hf_token=None, persist_directory: str = "chroma_db"):
        """
        Khởi tạo RAG với API key.

        Args:
            api_key (str): Groq API key
            hf_token (str): HuggingFace token
            persist_directory (str): Thư mục để lưu/tải ChromaDB.
        """
        if not api_key:
            raise ValueError("❌ Vui lòng cung cấp GROQ_API_KEY.")
        if not hf_token:
            raise ValueError("❌ Vui lòng cung cấp HF_TOKEN.")

        # FIX: truyền key trực tiếp, không ghi os.environ toàn cục
        self.llm = ChatGroq(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=TEMPERATURE,
            # Thêm giới hạn retry để tránh LangSmith hiển thị treo 10-20s khi bị rate limit
            max_retries=1, 
        )

        # FIX: truyền token trực tiếp vào HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.persist_directory = persist_directory
        self.vectorstore = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.generation_chain = None
        self.doc_count = 0
        self.chat_history = [] # Lưu bộ nhớ hội thoại

        print("✅ Đã khởi tạo SimpleRAG thành công!")

    def _build_prompt(self) -> PromptTemplate:
        template = """Bạn là nhân viên CSKH chuyên nghiệp, tự tin. Hãy trả lời tự nhiên bằng tiếng Việt và tuyệt đối không để lộ việc bạn đang đọc tài liệu.

HƯỚNG DẪN QUAN TRỌNG:
1. Trả lời trực tiếp vào trọng tâm. TUYỆT ĐỐI KHÔNG mở đầu bằng "Theo tài liệu...", "Dựa vào ngữ cảnh...".
2. Nếu đáp án là "Có" hoặc "Không", phải giải thích rõ lý do và điều kiện áp dụng (nếu có) dựa theo ngữ cảnh.
3. 'Lịch sử trò chuyện' chỉ để làm rõ đại từ ("nó", "cái đó"). Bỏ qua lịch sử nếu câu hỏi sang chủ đề mới.

Lịch sử trò chuyện:
{chat_history}

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
        return PromptTemplate(template=template, input_variables=["chat_history", "context", "question"])

    def load_documents(self, documents: list[str]):
        """
        Nạp tài liệu vào RAG.

        Args:
            documents (list[str]): Danh sách văn bản thô

        Returns:
            self
        """
        print(f"📚 Đang xử lý {len(documents)} tài liệu...")

        bm25_path = Path(self.persist_directory) / "bm25_index.pkl"

        # --- KIỂM TRA LƯU TRỮ (Lưu trữ BM25 và Vector) ---
        if Path(self.persist_directory).is_dir() and bm25_path.exists():
            print(f"  → Đang tải Vector database từ '{self.persist_directory}' và BM25 từ '{bm25_path}'...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            with open(bm25_path, "rb") as f:
                self.bm25_retriever = pickle.load(f)
            
            try:
                self.doc_count = self.vectorstore._collection.count()
            except Exception:
                self.doc_count = len(documents)
        else:
            print("  → Đang tạo và lưu Vector database & BM25...")
            print("     (Quá trình này có thể mất vài phút, các lần sau sẽ nhanh hơn)")
            
            # FIX: chunk TRƯỚC, tokenize SAU
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""],
                is_separator_regex=False,
            )
            raw_chunks = text_splitter.create_documents(documents)
            self.doc_count = len(raw_chunks)
            print(f"  → Đã chia thành {self.doc_count} đoạn nhỏ")

            # Tokenize từng chunk bằng pyvi
            vector_docs = []
            bm25_docs = []
            for chunk in raw_chunks:
                tokenized_content = tokenize(chunk.page_content)
                metadata = chunk.metadata.copy()
                metadata["original_content"] = chunk.page_content
                
                vector_docs.append(Document(
                    page_content=f"passage: {tokenized_content}",
                    metadata=metadata,
                ))
                
                bm25_docs.append(Document(
                    page_content=tokenized_content,
                    metadata=metadata,
                ))

            try:
                self.vectorstore = Chroma.from_documents(
                    vector_docs, self.embeddings, persist_directory=self.persist_directory
                )
            except Exception as e:
                print(f"❌ Đã xảy ra lỗi khi tạo vector database: {e}")
                print("   Vui lòng kiểm tra lại tài liệu đầu vào hoặc tài nguyên hệ thống.")
                return self # Dừng lại nếu không tạo được vectorstore

            # Khởi tạo và Lưu BM25
            self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
            self.bm25_retriever.k = K_RETRIEVAL
            
            with open(bm25_path, "wb") as f:
                pickle.dump(self.bm25_retriever, f)
            print(f"  → Đã lưu BM25 xuống '{bm25_path}'")

        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": K_RETRIEVAL,
                "score_threshold": 0.3,
            },
        )

        # FIX: dùng LCEL, tách riêng chuỗi sinh câu trả lời để có thể tái sử dụng
        prompt = self._build_prompt()

        self.generation_chain = prompt | self.llm | StrOutputParser()

        print(f"✅ Đã nạp xong {self.doc_count} đoạn tài liệu vào RAG!")
        return self

    def ask(self, question: str, show_context: bool = False, return_context: bool = False, use_history: bool = True) -> str | dict:
        """
        Đặt câu hỏi cho chatbot.

        Args:
            question (str): Câu hỏi
            show_context (bool): Có in ra các đoạn tài liệu được dùng không?
            return_context (bool): Nếu True, trả về dict chứa 'answer' và 'context'
            use_history (bool): Có sử dụng và lưu lại lịch sử trò chuyện hay không?

        Returns:
            str | dict: Câu trả lời hoặc dict chứa câu trả lời và ngữ cảnh
        """
        if not self.generation_chain or not self.vector_retriever or not self.bm25_retriever:
            return "❌ Chưa có tài liệu. Hãy gọi load_documents() trước."

        # Format lịch sử trò chuyện
        history_str = ""
        if use_history and self.chat_history:
            history_str = "\n".join([f"Người dùng: {q}\nAI: {a}" for q, a in self.chat_history])
        else:
            history_str = "Chưa có."

        # Thêm bước Query Rewrite (Mở rộng truy vấn) để giải quyết vấn đề vocabulary mismatch
        rewrite_prompt = PromptTemplate(
            template="""You are an expert at retail policies.
Your task is to rewrite the user's query into a clear, standalone set of search keywords for retrieving policy documents.
1. Resolve pronouns: If the query contains pronouns like "nó", "sản phẩm đó", refer back to the Recent Chat History and replace them with the actual product/entity name.
2. Add categories: CRITICAL: If the user mentions a specific product (e.g., 'áo lót', 'tủ lạnh', 'son môi'), you MUST explicitly add its broader policy category (e.g., 'trang phục lót', 'hàng điện tử', 'mỹ phẩm') to the keywords. Keep the original words as well.
Just output the keywords in Vietnamese. Do not answer the question.

Recent Chat History:
{chat_history}

Original Query: {question}
Keywords:""",
            input_variables=["chat_history", "question"]
        )
        try:
            rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()
            enhanced_question = rewrite_chain.invoke({"chat_history": history_str, "question": question})
            if show_context:
                print(f"🔍 Câu hỏi gốc: {question}")
                print(f"🔍 Truy vấn mở rộng: {enhanced_question}")
        except Exception as e:
            enhanced_question = question
            if show_context:
                print(f"⚠️ Lỗi Query Rewrite, dùng câu gốc: {e}")

        # FIX: Tokenize câu hỏi bằng pyvi ĐỂ ĐỒNG BỘ với tài liệu đã lưu
        tokenized_query = tokenize(enhanced_question)

        # 1. Retrieve: Lấy ngữ cảnh song song
        # Vector cần prefix 'query: '
        vector_docs = self.vector_retriever.invoke(f"query: {tokenized_query}")
        # BM25 chỉ cần câu hỏi đã tokenize
        bm25_docs = self.bm25_retriever.invoke(tokenized_query)

        # Gộp kết quả bằng Reciprocal Rank Fusion (RRF)
        # FIX: Dùng original_content từ metadata làm khóa gộp (dedup key)
        # vì page_content khác nhau giữa Vector ("passage: ...") và BM25 ("...")
        doc_scores = {}
        c = 60 # Hằng số RRF
        
        def _get_dedup_key(doc):
            """Lấy khóa gộp thống nhất cho cả Vector doc và BM25 doc."""
            return doc.metadata.get("original_content", doc.page_content)

        def _get_clean_context(doc):
            """Lấy nội dung gốc sạch để gửi cho LLM, có fallback an toàn."""
            original = doc.metadata.get("original_content")
            if original:
                return original
            # Fallback: loại bỏ prefix "passage: " và thay dấu gạch dưới
            text = doc.page_content
            if text.startswith("passage: "):
                text = text[len("passage: "):]
            return text.replace("_", " ")

        for rank, doc in enumerate(vector_docs):
            key = _get_dedup_key(doc)
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 0.6 * (1 / (rank + c))
            
        for rank, doc in enumerate(bm25_docs):
            key = _get_dedup_key(doc)
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0.0}
            doc_scores[key]["score"] += 0.4 * (1 / (rank + c))

        # Sắp xếp theo điểm và lấy K_RETRIEVAL tài liệu tốt nhất
        ranked_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        docs = [item["doc"] for item in ranked_docs[:K_RETRIEVAL]]

        if show_context:
            print(f"\n📄 Đoạn tài liệu được dùng ({len(docs)}):")
            for i, doc in enumerate(docs, 1):
                print(f"  [{i}] {doc.page_content[:150]}...")

        # 2. Generate: Sinh câu trả lời từ ngữ cảnh
        # FIX: Dùng _get_clean_context để đảm bảo LLM luôn nhận văn bản sạch
        # (không có prefix "passage:", không có dấu gạch dưới từ pyvi)
        context_parts = [_get_clean_context(doc) for doc in docs]
        context_str = "\n\n".join(context_parts)
        answer = self.generation_chain.invoke({
            "chat_history": history_str,
            "context": context_str,
            "question": question  # Dùng câu hỏi gốc, không có prefix
        })

        if use_history:
            # Lưu vào bộ nhớ hội thoại (giữ lại 3 lượt gần nhất)
            self.chat_history.append((question, answer))
            if len(self.chat_history) > 3:
                self.chat_history.pop(0)

        if return_context:
            return {"answer": answer, "context": docs}

        return answer

    def ask_batch(self, questions: list[str]) -> dict:
        """
        Hỏi nhiều câu hỏi cùng lúc.

        Args:
            questions (list[str]): Danh sách câu hỏi

        Returns:
            dict: {câu_hỏi: câu_trả_lời}
        """
        if not self.generation_chain or not self.vector_retriever or not self.bm25_retriever:
            return {q: "❌ Chưa có tài liệu. Hãy gọi load_documents() trước." for q in questions}
        return {q: self.ask(q) for q in questions}

    def get_stats(self) -> dict:
        """Lấy thông tin thống kê về RAG."""
        return {
            "doc_count": self.doc_count,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "temperature": TEMPERATURE,
            "k_retrieval": K_RETRIEVAL,
            "model": MODEL_NAME,
            "retriever": "Hybrid BM25 (40%) + Vector (60%)",
            "embedding_model": "BAAI/bge-m3",
        }
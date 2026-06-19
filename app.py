import streamlit as st
import os
import sys
import uuid

# Thêm src vào đường dẫn để import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_utils import SimpleRAG
from src.doc_utils import Converter
from dotenv import load_dotenv

# Load các biến môi trường từ file .env
load_dotenv()

# Cấu hình trang web
st.set_page_config(page_title="RAG Chatbot", page_icon="🛒", layout="wide")

DATA_DIR = "data"

@st.cache_resource(show_spinner=False)
def init_rag_system():
    """Khởi tạo RAG System và nạp tài liệu (Chỉ chạy 1 lần)"""
    api_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    
    if not api_key or not hf_token:
        st.error("⚠️ Thiếu API Key. Vui lòng kiểm tra file .env (cần GROQ_API_KEY và HF_TOKEN)")
        st.stop()
        
    rag = SimpleRAG(api_key, hf_token)
    converter = Converter()
    
    # Đọc dữ liệu từ thư mục
    docs = converter.load_documents_from_folder(DATA_DIR)
    if docs:
        rag.load_documents(docs)
    else:
        st.warning(f"⚠️ Không tìm thấy tài liệu trong thư mục {DATA_DIR}/")
        
    return rag

st.title("🛒 RAG Chatbot")
st.markdown("Hỏi đáp thông minh dựa trên chính sách và sản phẩm.")

# Khởi tạo hệ thống
try:
    with st.spinner("Đang khởi tạo hệ thống RAG và nạp tài liệu..."):
        rag = init_rag_system()
except Exception as e:
    import traceback
    print(f"❌ Lỗi khởi tạo hệ thống RAG: {e}")
    traceback.print_exc()
    st.error("⚠️ Đã xảy ra lỗi khi khởi tạo hệ thống. Vui lòng kiểm tra lại cấu hình hoặc thử lại sau.")
    st.stop()

# Khởi tạo state cho đa luồng chat (sessions)
if "sessions" not in st.session_state:
    default_session_id = str(uuid.uuid4())
    st.session_state.sessions = {default_session_id: []}
    st.session_state.current_session = default_session_id
    st.session_state.session_names = {default_session_id: "Đoạn chat mới"}

def sync_rag_history():
    """Đồng bộ lịch sử chat của phiên hiện tại vào đối tượng RAG trước khi truy vấn."""
    if not rag: return
    rag.chat_history = []
    msgs = st.session_state.sessions[st.session_state.current_session]
    user_msg = None
    for m in msgs:
        if m["role"] == "user":
            user_msg = m["content"]
        elif m["role"] == "assistant" and user_msg:
            rag.chat_history.append((user_msg, m["content"]))
            user_msg = None
    # Chỉ giữ lại 3 lượt gần nhất để tránh vượt quá token
    rag.chat_history = rag.chat_history[-3:]

# Giao diện Sidebar (Bảng bên trái)
with st.sidebar:
    st.header("💬 Cuộc trò chuyện")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        new_session_id = str(uuid.uuid4())
        st.session_state.sessions[new_session_id] = []
        st.session_state.session_names[new_session_id] = "Đoạn chat mới"
        st.session_state.current_session = new_session_id
        st.rerun()

    st.divider()
    
    st.subheader("Lịch sử")
    for session_id, name in reversed(list(st.session_state.session_names.items())):
        is_active = session_id == st.session_state.current_session
        button_label = f"💬 {name}" if not is_active else f"👉 {name}"
        if st.button(button_label, key=f"btn_{session_id}", use_container_width=True):
            st.session_state.current_session = session_id
            st.rerun()
            
    st.divider()
    if st.button("🗑️ Xóa toàn bộ", use_container_width=True):
        default_session_id = str(uuid.uuid4())
        st.session_state.sessions = {default_session_id: []}
        st.session_state.current_session = default_session_id
        st.session_state.session_names = {default_session_id: "Đoạn chat mới"}
        st.rerun()

# Quản lý trạng thái tin nhắn
current_msgs = st.session_state.sessions[st.session_state.current_session]

# Hiển thị lịch sử chat
for message in current_msgs:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input của người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Cập nhật tên phiên chat nếu đây là câu hỏi đầu tiên
    if not current_msgs:
        title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        st.session_state.session_names[st.session_state.current_session] = title
        
    # Hiển thị câu hỏi
    current_msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tạo và hiển thị câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            try:
                sync_rag_history()
                response = rag.ask(prompt, use_history=True)
            except Exception as e:
                import traceback
                print(f"❌ Lỗi hệ thống khi xử lý câu hỏi: {e}")
                traceback.print_exc()
                response = "❌ Đã xảy ra lỗi trong quá trình xử lý câu hỏi. Vui lòng thử lại sau."
            st.markdown(response)
            
    # Lưu vào lịch sử
    current_msgs.append({"role": "assistant", "content": response})
    st.rerun() # Refresh lại để hiển thị tiêu đề chat trên sidebar
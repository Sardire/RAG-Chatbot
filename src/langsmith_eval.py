"""
LangSmith Evaluation Module
Evaluate RAG chatbot using LangSmith's evaluation framework

Sử dụng Groq (miễn phí) thay vì OpenAI để tiết kiệm chi phí
"""
import os
import sys
import time
import re
import json
from typing import Optional, Callable, Dict, Any, List
from functools import wraps

# Add parent directory to path to allow imports when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsmith import Client, evaluate, EvaluationResult
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag_utils import SimpleRAG

# Try to import load_evaluator with fallback for different versions
try:
    from langchain_classic.evaluation import load_evaluator
except ImportError:
    try:
        from langchain_community.evaluation import load_evaluator
    except ImportError:
        try:
            from langchain.evaluation import load_evaluator
        except ImportError:
            print("⚠️  load_evaluator not available. Evaluation functions may not work correctly.")


def retry_with_backoff(max_retries: int = 4, backoff_factor: float = 2):
    """Decorator để retry với exponential backoff khi gặp rate limit"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    # Xử lý bao quát các thông báo lỗi rate limit của Groq API
                    is_rate_limit = any(term in error_msg for term in [
                        "rate limit", "rate_limit", "429", "too many requests"
                    ])
                    
                    if attempt < max_retries - 1:
                        if is_rate_limit:
                            wait_time = 30 * (backoff_factor ** attempt) # Tăng thời gian chờ vì Groq reset sau mỗi phút
                            print(f"⏳ Rate limit hit. Chờ {wait_time}s trước khi retry (attempt {attempt + 1}/{max_retries})...")
                        else:
                            wait_time = 2 # Nếu là lỗi parse JSON hoặc rớt mạng, chờ 2s rồi thử lại
                            print(f"⚠️ Lỗi {e}. Thử lại sau {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        if is_rate_limit:
                            print(f"❌ Rate limit exceeded after {max_retries} attempts")
                        else:
                            print(f"❌ Failed after {max_retries} attempts: {e}")
                        raise
            return None
        return wrapper
    return decorator


class RAGEvaluator:
    """
    Đánh giá chatbot RAG sử dụng LangSmith
    
    ✓ Dùng Groq cho LLM-as-Judge (miễn phí, chính xác cao)
    ✓ Không cần OpenAI API key
    ✓ Hỗ trợ cả legacy và custom evaluators
    """
    
    def __init__(self, rag_system: SimpleRAG, groq_api_key: Optional[str] = None):
        """
        Khởi tạo evaluator
        
        Args:
            rag_system (SimpleRAG): Hệ thống RAG cần đánh giá
            groq_api_key (str): Groq API key (tuỳ chọn, nếu không sẽ lấy từ env)
        """
        self.rag_system = rag_system
        self.client = Client()
        self.context_cache = {} # Cache để truyền context ngầm vào evaluator
        
        # Setup Groq API key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Initialize Groq LLM for evaluations
        if os.environ.get("GROQ_API_KEY"):
            self.llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0.2,  # Cho evaluation, cần deterministic
                api_key=os.environ.get("GROQ_API_KEY"),
                timeout=60,  # Tăng timeout
                max_retries=1
            )
            print(f"✅ Groq LLM ('openai/gpt-oss-120b') initialized for evaluations")
        else:
            print("⚠️  GROQ_API_KEY không được thiết lập - LLM-as-Judge sẽ không hoạt động")
            self.llm = None
    
    def create_predict_wrapper(self) -> Callable:
        """
        Tạo wrapper function cho predict
        
        Returns:
            Callable: Function nhận input dict và trả về output dict
        """
        @retry_with_backoff(max_retries=3, backoff_factor=2)
        def predict_wrapper(inputs: dict) -> dict:
            """
            Bọc hàm predict để chuyển đổi định dạng cho LangSmith
            
            Args:
                inputs (dict): Input từ dataset, phải chứa 'question'
            
            Returns:
                dict: Output với key 'output'
            """
            # LangSmith có thể gửi input dưới dạng 'question' hoặc 'input'
            question = inputs.get("question") or inputs.get("input")
            
            if not question:
                return {"output": ""}
            
            # Thêm delay chủ động giữa các lần hỏi để giảm tải RPM và TPM
            time.sleep(3)
            time.sleep(1) # Giảm từ 3 giây xuống 1 giây
            
            try:
                # Lấy câu trả lời và ngữ cảnh từ RAG system 
                # Tắt lưu lịch sử (use_history=False) để các câu đánh giá hoàn toàn độc lập
                result = self.rag_system.ask(question, return_context=True, use_history=False)
                if isinstance(result, dict):
                    docs = result.get("context", [])
                    context_str = "\n\n".join([doc.page_content for doc in docs])
                    
                    # FIX: Lưu context vào RAM để truyền ngầm cho Evaluator
                    # Tránh bị dính vào mục 'Outputs' trên giao diện LangSmith
                    self.context_cache[question] = context_str
                    
                    return {"output": result.get("answer", "")}
                else:
                    self.context_cache[question] = ""
                    return {"output": result if result else ""}
            except Exception as e:
                self.context_cache[question] = ""
                raise e # Đẩy lỗi ra để decorator bắt và tiến hành retry
        
        return predict_wrapper
    

    def _evaluate_with_llm(self, prompt_template: str, inputs: dict) -> tuple[float, str]:
        """Hàm dùng chung để chạy evaluator với LLM tuỳ chỉnh, cho điểm 0.0 - 1.0 (bước 0.1)"""
        if not self.llm:
            return 0.0, "LLM not available"
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result_text = chain.invoke(inputs)
            import re
            match = re.search(r'(?:Score|Điểm số)\s*:\s*([0-9]+(?:\.[0-9]+)?)', result_text, re.IGNORECASE)
            
            score_val = 0.0
            if match:
                score_val = float(match.group(1))
            else:
                numbers = re.findall(r'\b(?:10|[0-9])\b', result_text)
                if numbers:
                    score_val = float(numbers[-1])
            
            normalized_score = round(min(max(score_val / 10.0, 0.0), 1.0), 1)
            reasoning = result_text.replace('\n', ' ')
            
            return normalized_score, reasoning[:250]
            
        except Exception as e:
            raise e

    @retry_with_backoff(max_retries=4, backoff_factor=2)
    def correctness_evaluator(self, run, example) -> EvaluationResult:
        """Đánh giá Độ Chính Xác (Correctness) so với đáp án mẫu"""
        try:
            question = example.inputs.get("question") or example.inputs.get("input", "")
            prediction = run.outputs.get("output", "")
            reference = (example.outputs.get("answer") or 
                        example.outputs.get("expected_output") or 
                        example.outputs.get("output", ""))

            if not prediction or not reference:
                return EvaluationResult(key="correctness", score=0.0, comment="Missing prediction or reference")

            prompt = """Bạn là giám khảo đánh giá AI. Hãy đánh giá ĐỘ CHÍNH XÁC (Correctness).
So sánh câu trả lời của AI (Prediction) với câu trả lời mẫu (Reference) cho câu hỏi (Question).

TIÊU CHÍ CHẤM ĐIỂM (Chọn 1 mức điểm phù hợp nhất):
- 0 điểm: Trả lời sai hoàn toàn, mâu thuẫn hoặc đi ngược lại với đáp án mẫu.
- 3 điểm: Có đề cập đến chủ đề nhưng phần lớn thông tin quan trọng bị sai lệch hoặc thiếu.
- 5 điểm: Trả lời đúng khoảng một nửa nội dung chính, nhưng vẫn thiếu sót nhiều ý quan trọng.
- 8 điểm: Trả lời đúng hầu hết các ý chính, chỉ thiếu một vài chi tiết nhỏ không ảnh hưởng lớn.
- 10 điểm: Trả lời chính xác, đầy đủ và hoàn toàn khớp với ý nghĩa của đáp án mẫu.

Question: {question}
Reference Answer: {reference}
AI Prediction: {prediction}

TRẢ VỜI THEO ĐÚNG ĐỊNH DẠNG SAU:
Reasoning: [Phân tích chi tiết câu trả lời so với tiêu chí để quyết định mức điểm]
Score: [Chỉ ghi một con số nguyên từ 0-10]"""
            
            score, reasoning = self._evaluate_with_llm(prompt, {
                "question": question,
                "reference": reference,
                "prediction": prediction
            })
            
            return EvaluationResult(key="correctness", score=score, comment=reasoning)
        except Exception as e:
            raise e

    @retry_with_backoff(max_retries=4, backoff_factor=2)
    def relevance_evaluator(self, run, example) -> EvaluationResult:
        """Đánh giá Độ Liên Quan (Relevance) với câu hỏi"""
        try:
            question = example.inputs.get("question") or example.inputs.get("input", "")
            prediction = run.outputs.get("output", "")

            if not prediction or not question:
                return EvaluationResult(key="relevance", score=0.0, comment="Missing prediction or question")

            prompt = """Bạn là giám khảo đánh giá AI. Hãy đánh giá ĐỘ LIÊN QUAN (Relevance).
Xem xét câu trả lời của AI (Prediction) có trả lời đúng trọng tâm câu hỏi (Question) không.

TIÊU CHÍ CHẤM ĐIỂM (Chọn 1 mức điểm phù hợp nhất):
- 0 điểm: Hoàn toàn lạc đề, không giải quyết được bất kỳ khía cạnh nào của câu hỏi.
- 3 điểm: Có nhắc đến từ khóa nhưng lan man, không thực sự trả lời đúng mục đích câu hỏi.
- 5 điểm: Giải quyết được một phần câu hỏi, nhưng phần còn lại chứa nhiều thông tin dư thừa hoặc chưa trúng đích.
- 8 điểm: Trả lời đi thẳng vào trọng tâm, nhưng vẫn còn chèn thêm một số thông tin phụ không cần thiết.
- 10 điểm: Trả lời trực tiếp, súc tích, hoàn toàn bám sát và giải quyết triệt để câu hỏi.

Question: {question}
AI Prediction: {prediction}

TRẢ VỜI THEO ĐÚNG ĐỊNH DẠNG SAU:
Reasoning: [Phân tích việc câu trả lời có trúng đích hay không dựa trên tiêu chí]
Score: [Chỉ ghi một con số nguyên từ 0-10]"""
            
            score, reasoning = self._evaluate_with_llm(prompt, {
                "question": question,
                "prediction": prediction
            })
            
            return EvaluationResult(key="relevance", score=score, comment=reasoning)
        except Exception as e:
            raise e

    @retry_with_backoff(max_retries=4, backoff_factor=2)
    def faithfulness_evaluator(self, run, example) -> EvaluationResult:
        """Đánh giá Độ Trung Thực (Faithfulness) dựa trên tài liệu retrieve được"""
        try:
            prediction = run.outputs.get("output", "")
            # Lấy context từ bộ nhớ cache ngầm dựa trên câu hỏi
            question = example.inputs.get("question") or example.inputs.get("input", "")
            context = self.context_cache.get(question, "")

            if not prediction:
                return EvaluationResult(key="faithfulness", score=0.0, comment="Missing prediction")
            if not context:
                return EvaluationResult(key="faithfulness", score=0.0, comment="Missing retrieved context")

            prompt = """Bạn là giám khảo đánh giá AI. Hãy đánh giá ĐỘ TRUNG THỰC (Faithfulness).
Xem xét câu trả lời của AI (Prediction) có trung thực và hoàn toàn dựa vào ngữ cảnh được cung cấp (Context) không.

TIÊU CHÍ CHẤM ĐIỂM (Chọn 1 mức điểm phù hợp nhất):
- 0 điểm: Hoàn toàn bịa đặt (Hallucination), mọi thông tin đều không có trong ngữ cảnh.
- 3 điểm: Dựa một phần rất nhỏ vào ngữ cảnh, phần lớn thông tin tự sáng tác hoặc dùng kiến thức bên ngoài.
- 5 điểm: Dựa vào ngữ cảnh nhưng suy diễn sai lệch ý hoặc tự thêm các chi tiết quan trọng không có thực.
- 8 điểm: Bám sát ngữ cảnh, có thể có từ ngữ diễn đạt khác nhưng không làm thay đổi bản chất thông tin.
- 10 điểm: 100% thông tin trong câu trả lời có thể được tìm thấy và trích xuất trực tiếp từ ngữ cảnh.

Context: {context}
AI Prediction: {prediction}

TRẢ VỜI THEO ĐÚNG ĐỊNH DẠNG SAU:
Reasoning: [Chỉ ra các thông tin có/không có trong ngữ cảnh để biện luận cho mức điểm]
Score: [Chỉ ghi một con số nguyên từ 0-10]"""
            
            score, reasoning = self._evaluate_with_llm(prompt, {
                "context": context,
                "prediction": prediction
            })
            
            return EvaluationResult(key="faithfulness", score=score, comment=reasoning)
        except Exception as e:
            raise e

    @retry_with_backoff(max_retries=4, backoff_factor=2)
    def combined_evaluator(self, run, example) -> list[dict]:
        """Đánh giá cả 3 tiêu chí trong 1 lần gọi LLM duy nhất để tối ưu tốc độ và API"""
        try:
            question = example.inputs.get("question") or example.inputs.get("input", "")
            prediction = run.outputs.get("output", "")
            reference = (example.outputs.get("answer") or 
                        example.outputs.get("expected_output") or 
                        example.outputs.get("output", ""))
            context = self.context_cache.get(question, "")

            if not prediction:
                return [{"key": "error", "score": 0.0, "comment": "Missing prediction"}]

            prompt_text = """Bạn là giám khảo đánh giá AI. Hãy chấm điểm câu trả lời của AI dựa trên 3 TIÊU CHÍ cốt lõi:

1. CORRECTNESS (Độ chính xác so với Reference): 0 (Sai hoàn toàn) -> 10 (Chính xác hoàn toàn).
2. RELEVANCE (Độ liên quan tới Question): 0 (Lạc đề) -> 10 (Đúng trọng tâm).
3. FAITHFULNESS (Độ trung thực với Context): 0 (Bịa đặt hoàn toàn) -> 10 (100% dựa vào Context).

NGỮ CẢNH ĐỂ ĐỐI CHIẾU:
Context: {context}
Question: {question}
Reference Answer: {reference}
AI Prediction: {prediction}

TRẢ VỜI BẮT BUỘC BẰNG JSON, KHÔNG CÓ BẤT KỲ VĂN BẢN NÀO KHÁC BÊN NGOÀI:
{{
    "correctness": {{"score": 8, "reasoning": "[Lý do ngắn gọn]"}},
    "relevance": {{"score": 10, "reasoning": "[Lý do ngắn gọn]"}},
    "faithfulness": {{"score": 5, "reasoning": "[Lý do ngắn gọn]"}}
}}"""
            
            prompt = PromptTemplate.from_template(prompt_text)
            chain = prompt | self.llm | StrOutputParser()
            
            result_text = chain.invoke({
                "context": context if context else "Không có ngữ cảnh.",
                "question": question,
                "reference": reference if reference else "Không có đáp án mẫu.",
                "prediction": prediction
            })
            
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                raise ValueError("LLM không trả về định dạng JSON")
                
            data = json.loads(json_match.group(0))
            results = []
            for key in ["correctness", "relevance", "faithfulness"]:
                if key in data:
                    score_val = float(data[key].get("score", 0))
                    normalized_score = round(min(max(score_val / 10.0, 0.0), 1.0), 1)
                    results.append({"key": key, "score": normalized_score, "comment": data[key].get("reasoning", "")})
            return results
        except Exception as e:
            raise e

    def run_evaluation(
        self,
        dataset_name: str = "RAG eval #1",
        experiment_prefix: str = "rag-evaluation",
        max_concurrency: int = 1,
        evaluators: Optional[List[Callable]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Chạy evaluation trên dataset từ LangSmith
        
        Sử dụng custom LLM-based evaluators (Thang 0.0 - 1.0, bước 0.1):
        - correctness: Độ chính xác so với đáp án mẫu
        - relevance: Đi đúng trọng tâm câu hỏi
        - faithfulness: Trung thực với ngữ cảnh (chống ảo giác)
        
        Args:
            dataset_name (str): Tên dataset trong LangSmith
            experiment_prefix (str): Prefix cho experiment name
            max_concurrency (int): Số lượng concurrent evaluations
            evaluators (list): List các evaluator functions (nếu không, sử dụng default)
        
        Returns:
            dict: Kết quả evaluation hoặc None nếu thất bại
        """
        print(f"\n🚀 Bắt đầu evaluation trên dataset '{dataset_name}'...")
        print(f"📊 Experiment prefix: {experiment_prefix}")
        print("🎯 Sử dụng 3 tiêu chí cốt lõi (Core RAG Triad)")
        
        # Create predict wrapper
        predict_fn = self.create_predict_wrapper()
        
        # Use provided evaluators or default ones
        if evaluators is None:
            evaluators = [self.combined_evaluator]
            print("✅ Sử dụng Combined Evaluator (Gộp 3 tiêu chí vào 1 lần gọi API):")
            print("   • Correctness, Relevance, Faithfulness")
        
        try:
            # Run evaluation
            results = evaluate(
                predict_fn,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=experiment_prefix,
                max_concurrency=max_concurrency,
                metadata={
                    "evaluator_version": "3.0-llm-based",
                    "evaluation_type": "LangChain Default Evaluators",
                    "llm_model": f"{self.llm.model_name} (Groq)" if self.llm else "heuristic_only",
                    "evaluators_count": len(evaluators)
                }
            )
            
            print(f"\n✅ Evaluation hoàn thành!")
            print(f"   Dataset: {dataset_name}")
            print(f"   Experiment: {experiment_prefix}")
            print(f"   Results available at: https://smith.langchain.com")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Lỗi trong quá trình evaluation: {e}")
            import traceback
            traceback.print_exc()
            print("\n💡 Troubleshooting tips:")
            print("   1. Check if dataset exists: client.list_datasets()")
            print("   2. Verify GROQ_API_KEY is set correctly")
            print("   3. Check internet connection")
            print("   4. Ensure LangChain evaluators are properly installed")
            return None
    
    def get_dataset_from_langsmith(self, dataset_name: str):
        """
        Lấy thông tin về dataset từ LangSmith
        
        Args:
            dataset_name (str): Tên dataset
        
        Returns:
            Dataset: LangSmith dataset object hoặc None
        """
        try:
            dataset = self.client.read_dataset(dataset_name=dataset_name)
            print(f"\n📊 Dataset '{dataset_name}':")
            print(f"   • ID: {dataset.id}")
            print(f"   • Name: {dataset.name}")
            print(f"   • Description: {dataset.description}")
            print(f"   • Created: {dataset.created_at}")
            
            # Count examples
            examples = list(self.client.list_examples(dataset_id=dataset.id))
            print(f"   • Number of examples: {len(examples)}")
            
            # Show first example structure for debugging
            if examples:
                print(f"\n   📝 First example structure:")
                print(f"      Inputs: {list(examples[0].inputs.keys())}")
                print(f"      Outputs: {list(examples[0].outputs.keys())}")
            
            return dataset
        except Exception as e:
            print(f"❌ Không thể lấy dataset '{dataset_name}': {e}")
            print("\n   💡 To create a dataset:")
            print("   from langsmith import Client")
            print("   client = Client()")
            print('   dataset = client.create_dataset("RAG eval #1")')
            print('   client.create_examples(')
            print('       inputs=[{"question": "What is X?"}],')
            print('       outputs=[{"expected_output": "X is Y"}],')  # Note: use expected_output
            print('       dataset_id=dataset.id')
            print('   )')
            return None
    
    def list_available_datasets(self):
        """
        Liệt kê tất cả datasets có sẵn trong LangSmith
        """
        try:
            datasets = list(self.client.list_datasets())
            if not datasets:
                print("\n📭 No datasets found in LangSmith")
                return []
            
            print(f"\n📚 Available datasets ({len(datasets)}):")
            for i, dataset in enumerate(datasets, 1):
                examples = list(self.client.list_examples(dataset_id=dataset.id))
                examples_count = len(examples)
                print(f"   {i}. {dataset.name} ({examples_count} examples)")
                
                # Show structure of first example if exists
                if examples:
                    print(f"      Input keys: {list(examples[0].inputs.keys())}")
                    print(f"      Output keys: {list(examples[0].outputs.keys())}")
            
            return datasets
        except Exception as e:
            print(f"❌ Error listing datasets: {e}")
            return []


def evaluate_rag_chatbot(
    rag_system: SimpleRAG,
    dataset_name: str = "RAG eval #1",
    experiment_prefix: str = "rag-evaluation",
    groq_api_key: Optional[str] = None,
    list_datasets_first: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Hàm tiện ích để đánh giá chatbot RAG
    
    Args:
        rag_system (SimpleRAG): Hệ thống RAG cần đánh giá
        dataset_name (str): Tên dataset trong LangSmith
        experiment_prefix (str): Prefix cho experiment name
        groq_api_key (str): Groq API key (tuỳ chọn)
        list_datasets_first (bool): Liệt kê datasets trước khi chạy evaluation
    
    Returns:
        dict: Kết quả evaluation hoặc None nếu thất bại
    """
    print("="*60)
    print("🎯 RAG Chatbot Evaluation with LangSmith")
    print("="*60)
    
    # Initialize evaluator (using Groq - free)
    evaluator = RAGEvaluator(rag_system, groq_api_key)
    
    # Optionally list available datasets
    if list_datasets_first:
        evaluator.list_available_datasets()
    
    # Get dataset information
    dataset = evaluator.get_dataset_from_langsmith(dataset_name)
    if dataset is None:
        print("\n⚠️  Cannot proceed with evaluation without dataset")
        return None
    
    # Ask for confirmation
    print(f"\n🤖 Ready to evaluate RAG system: {rag_system.__class__.__name__}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🏷️  Experiment: {experiment_prefix}")
    
    # Run evaluation
    results = evaluator.run_evaluation(
        dataset_name=dataset_name,
        experiment_prefix=experiment_prefix,
        max_concurrency=3  # Tăng lên 3 để chấm song song 3 câu hỏi cùng lúc
    )
    
    return results


# Example usage
if __name__ == "__main__":
    # This is just an example - actual usage will be from main.py
    print("LangSmith Evaluation Module")
    print("Import this module and call evaluate_rag_chatbot()")
    print("\nExample usage:")
    print("""
from src.langsmith_eval import evaluate_rag_chatbot
from src.rag_utils import SimpleRAG

# Initialize your RAG system
rag = SimpleRAG(chroma_path="./chroma_db")

# Run evaluation
results = evaluate_rag_chatbot(
    rag_system=rag,
    dataset_name="RAG eval #1",
    experiment_prefix="my-experiment",
    groq_api_key="your-groq-api-key",
    list_datasets_first=True
)
    """)
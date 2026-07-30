
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.document_manager import DocumentManager

class CustomDocumentManager(DocumentManager):
    def __init__(self, generator_type="deepseek", local_generator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator_type = generator_type
        self.local_generator = local_generator
        self.captured_questions = []

    def _generate_enhanced_questions(self, content: str, num_questions: int = 3):
        questions = []
        
        if self.generator_type == "deepseek":
            print("DEBUG: Calling super()._generate_enhanced_questions for DeepSeek")
            questions = super()._generate_enhanced_questions(content, num_questions)
            print(f"DEBUG: DeepSeek returned {len(questions)} questions")
        
        # Capture first 3 examples
        if len(self.captured_questions) < 3 and questions:
             self.captured_questions.append({
                 "content_snippet": content[:100] + "...",
                 "questions": questions
             })
        
        return questions

def test_integration():
    print("Testing DeepSeek Integration via CustomDocumentManager...")
    api_key = os.getenv("DEEPSEEK_API")
    if not api_key:
        print("Error: DEEPSEEK_API not found.")
        return

    dm = CustomDocumentManager(generator_type="deepseek")
    content = "DeepSeek (深度求索) is an artificial intelligence company based in China. It aims to develop AGI."
    
    print(f"Generating questions for content: {content[:50]}...")
    questions = dm._generate_enhanced_questions(content, num_questions=2)
    
    print("\nResult:")
    print(f"Questions: {questions}")
    print(f"Captured: {len(dm.captured_questions)}")
    if dm.captured_questions:
        print(f"Captured Sample: {dm.captured_questions[0]}")

if __name__ == "__main__":
    test_integration()

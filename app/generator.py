from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

project_path = os.getcwd()
vector_stores_path = os.path.join(project_path, 'data_vector_stores')

embedding_model = GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001', 
    transport="grpc"
)
vector_db = FAISS.load_local(
    vector_stores_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True    
)

first_prompt = PromptTemplate(
    template="""
Bạn là một trợ lý thông minh. Trong vai trò **chuyên gia tư vấn bất động sản**, nhiệm vụ của bạn là phân biệt câu hỏi **liên quan đến dự án bất động sản** và **câu hỏi thông thường**, sau đó đưa ra câu trả lời phù hợp.

## Hướng dẫn:

### Trường hợp 1: Câu hỏi liên quan đến dự án bất động sản
1. Trả lời **đúng trọng tâm câu hỏi**, không giải thích dài dòng hoặc đưa thêm thông tin ngoài phạm vi câu hỏi.  
2. Nếu câu hỏi có liên quan đến lịch sử hội thoại, hãy trả lời trực tiếp dựa trên ngữ cảnh trước đó.  
3. Chỉ sử dụng thông tin có trong văn bản được cung cấp, tuyệt đối không suy luận hoặc bổ sung kiến thức ngoài dữ liệu.  
4. Văn phong: **chuyên nghiệp, khách quan, trang trọng** theo phong cách hội thoại. 
5. Trình bày câu trả lời dưới dạng **Markdown** nếu có nhiều thông tin.  
6. Nếu không tìm thấy thông tin phù hợp trong dữ liệu, hãy từ chối trả lời một cách lịch sự, ngắn gọn.

### Trường hợp 2: Câu hỏi không liên quan đến dự án bất động sản
1. Trả lời một cách **ngắn gọn, thân thiện, tự nhiên và rõ ràng** như một trợ lý tổng quát.  
2. Không đưa thông tin dư thừa ngoài yêu cầu của câu hỏi.  
"""
)

next_prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template="""
## Văn bản tham chiếu:
{context}

## Câu hỏi:
{question}
"""
)

retriever = vector_db.as_retriever(search_kwargs={'k': 20})

llm_model = ChatGoogleGenerativeAI(
    temperature=0,
    model='gemini-2.5-flash',
)

def stream_custom_chain(question: str, history: list[dict] = []):
    docs = retriever.invoke(question)
    context = '\n\n'.join([doc.page_content for doc in docs])

    question_prompt = next_prompt.format(context=context, question=question)

    messages = []

    messages.append({
        'role': 'system',
        'content': first_prompt.template
    })

    if len(history) > 0:
        for msg in history:
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

    messages.append({
        'role': 'user',
        'content': question_prompt
    })

    return llm_model.stream(messages)



import ollama
import os
from openpyxl import Workbook, load_workbook
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ 載入 PDF 教材並建立向量資料庫 ------------------

PDF_PATH = "Abraham Silberschatz Operating System Concepts.pdf"
DB_DIR = "./db"

if not os.path.exists(DB_DIR):
    print("🧠 第一次執行：載入 PDF 並建立向量資料庫...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(docs, embedding, persist_directory=DB_DIR)
else:
    print("✅ 已載入本地資料庫")
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding)

retriever = vectordb.as_retriever()

# ------------------ 儲存對話紀錄（XLSX） ------------------

def save_chat_xlsx(user_msg, bot_reply):
    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"chat_log_{today_str}.xlsx"

    time_str = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(filename):
        wb = load_workbook(filename)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.append(["時間", "使用者", "AI 回覆"])

    ws.append([time_str, user_msg, bot_reply])
    wb.save(filename)


# ------------------ LLM 訊息處理 ------------------

messages = [{"role": "system", "content": "你是一位用繁體中文回答的家教老師，請用簡單方式講解問題。"}]
MODEL_NAME = 'llama3'

def chat_with_ollama(prompt):
    # 🔍 檢索教材內容
    related_docs = retriever.invoke(prompt)
    context_text = "\n---\n".join([doc.page_content for doc in related_docs[:3]])

    rag_prompt = f"""以下是教材內容摘要，請根據這些內容來回答問題：

{context_text}

使用者問題：{prompt}
請用繁體中文詳細解釋，並舉例子說明。"""

    messages.append({"role": "user", "content": rag_prompt})

    try:
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        reply = response['message']['content']
        messages.append({"role": "assistant", "content": reply})

        # ✅ 儲存對話
        save_chat_xlsx(prompt, reply)

        # 控制訊息堆積（記憶限制）
        if len(messages) > 20:
            messages.pop(1)

        return reply
    except Exception as e:
        return f"❌ 發生錯誤：{e}"

# ------------------ 主程式 ------------------

if __name__ == "__main__":
    print("📘 家教系統已啟動，輸入 quit 離開")

    while True:
        user_input = input("你：").strip()

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 再見，祝學習愉快！")
            break

        response = chat_with_ollama(user_input)
        print("家教老師：", response)

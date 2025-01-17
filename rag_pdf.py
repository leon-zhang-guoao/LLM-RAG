import gradio as gr
import os
import re
import uuid
import time
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import fitz
from PIL import Image
import chromadb
from smolagents import Tool, CodeAgent, LiteLLMModel
import google.generativeai as genai
import tiktoken
from dotenv import load_dotenv

os.environ["GEMINI_API_KEY"] = "key"
os.environ["DEEPSEEK_API_KEY"] = "key"

# 配置 API 密钥
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # 使用 GOOGLE_API_KEY
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 配置 Gemini API
genai.configure(api_key=GOOGLE_API_KEY)  # 使用 GOOGLE_API_KEY

# Tokenizer（用于检查文本长度）
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")

# 自定义 Gemini Embeddings 类
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为向量列表。"""
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(model="models/text-embedding-004", content=text)
                embeddings.append(result['embedding'])
                time.sleep(60 / 1500)  # 速率限制
            return embeddings
        except Exception as e:
            print(f"Gemini Embeddings API 请求失败: {e}")
            return []

    def embed_query(self, text: str) -> list[float]:
        """将单个文本转换为向量。"""
        try:
            result = genai.embed_content(model="models/text-embedding-004", content=text)
            return result['embedding']
        except Exception as e:
            print(f"Gemini Embeddings API 请求失败: {e}")
            return []

# PDF 页面渲染函数
def render_pdf_page(file, page_number=0):
    doc = fitz.open(file.name)
    page = doc[page_number]
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    return image

def render_file(file):
    return render_pdf_page(file, app.N)

def render_first(file):
    return render_pdf_page(file, 0), []

# Gradio 回调函数
def set_apikey(api_key: str):
    app.OPENAI_API_KEY = api_key
    return gr.Textbox.update(value='DS API key is Set', interactive=False)

def enable_api_box():
    return gr.Textbox.update(value=None, placeholder='Upload your DS API key', interactive=True)

def add_text(history, text: str):
    if not text:
        raise gr.Error('Enter text')
    # 将字典列表转换为列表的列表
    history = history + [[text, ""]]  # 用户消息和空的助手消息
    return history

def get_response(history, query, file):
    if not file:
        raise gr.Error(message='Upload a PDF')
    try:
        chain = app(file.name)
        result = chain({"question": query, 'chat_history': app.chat_history}, return_only_outputs=True)
        app.chat_history += [(query, result["answer"])]
        app.N = list(result['source_documents'][0])[1][1]['page']
        # 将助手消息添加到 history 中
        history[-1][1] = result["answer"]  # 更新最后一个消息的助手部分
        yield history, ''
    except Exception as e:
        print(f"Error in get_response: {e}")
        history[-1][1] = f"An error occurred: {str(e)}"  # 更新错误消息
        yield history, ''

# 自定义工具类
class my_app(Tool):
    def __init__(self, OPENAI_API_KEY: str = None) -> None:
        super().__init__()
        self.OPENAI_API_KEY: str = OPENAI_API_KEY
        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0
        self.description = "This is a custom tool for handling PDF files and conversational retrieval."
        self.name = "PDF Conversational Tool"
        self.inputs = {
            "file": {"description": "The PDF file to process", "type": "string"}
        }
        self.output_type = "any"

    def forward(self, file: str):
        if self.count == 0:
            self.chain = self.build_chain(file)
            self.count += 1
        return self.chain

    def process_file(self, file: str):
        loader = PyPDFLoader(file)
        documents = loader.load()
        file_name = os.path.basename(file)
        return documents, file_name

    def build_chain(self, file: str):
        documents, file_name = self.process_file(file)

        # 使用自定义的 GeminiEmbeddings
        embeddings = GeminiEmbeddings(api_key=GOOGLE_API_KEY)

        # 将文档和 Embeddings 存储到 Chroma
        pdfsearch = Chroma.from_documents(
            documents,
            embeddings,
            collection_name=file_name,
        )

        # 构建问答链
        chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model='deepseek-chat', openai_api_key=self.OPENAI_API_KEY, openai_api_base='https://api.deepseek.com', max_tokens=2048),
            retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
        )
        return chain

# 初始化应用
app = my_app(OPENAI_API_KEY=DEEPSEEK_API_KEY)
deepseek_model = LiteLLMModel(model_id="deepseek/deepseek-chat", api_key=DEEPSEEK_API_KEY)

agent = CodeAgent(
    tools=[app],
    additional_authorized_imports=['datetime', 'queue', 'statistics', 're', 'math', 'random', 'unicodedata', 'collections', 'time', 'itertools', 'stat', 'matplotlib', 'pandas', 'langchain'],
    model=deepseek_model
)

# Gradio 界面
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            with gr.Column(scale=8):
                api_key = gr.Textbox(placeholder='Enter DS API key', show_label=False, interactive=True, label="API Key")
            with gr.Column(scale=2):
                change_api_key = gr.Button('Change Key', variant="primary")
        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot', label="Chatbot")
            show_img = gr.Image(label='Upload PDF', height=680)
    with gr.Row():
        with gr.Column(scale=6):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter", label="Input")
        with gr.Column(scale=2):
            submit_btn = gr.Button('Submit', variant="primary")
        with gr.Column(scale=2):
            btn = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"], variant="primary")

    api_key.submit(
        fn=set_apikey,
        inputs=[api_key],
        outputs=[api_key]
    )
    change_api_key.click(
        fn=enable_api_box,
        outputs=[api_key]
    )
    btn.upload(
        fn=render_first,
        inputs=[btn],
        outputs=[show_img, chatbot]
    )
    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False
    ).success(
        fn=get_response,
        inputs=[chatbot, txt, btn],
        outputs=[chatbot, txt]
    ).success(
        fn=render_file,
        inputs=[btn],
        outputs=[show_img]
    )

# 启动应用
demo.queue()
demo.launch(server_name="host")
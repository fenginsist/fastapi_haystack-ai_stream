from fastapi import FastAPI, Query
from haystack import Document
from fastapi.responses import StreamingResponse
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from typing import AsyncGenerator
from haystack.utils import Secret
from haystack import component
from typing import Optional
from haystack.dataclasses import StreamingChunk

app = FastAPI()

import asyncio

stream_queue = asyncio.Queue()

# 初始化文档存储 并 写入文档
document_store = InMemoryDocumentStore()
documents = [Document(content="Haystack is an open-source NLP framework."),
             Document(content="FastAPI is a modern web framework for Python."),
             Document(content="Streaming responses can improve real-time user experiences.")
]
document_store.write_documents(documents=documents)

# check 文档存储中的文档
all_docs = document_store.count_documents()
if all_docs == len(documents):
    print("document size:", all_docs)
    print("All documents were successfully written to the document store.")
else:
    print("document size:", all_docs)
    print("Not all documents were written to the document store.")

# 初始化检索器 
retriever = InMemoryBM25Retriever(document_store=document_store)

# 自定义转换节点
@component
class DocumentToPrompt:
    @component.output_types(prompt=str)
    def run(self, documents: list[Document], query: Optional[str]="你是谁"):
        # 将文档内容拼接为一个提示字符串
        prompt = "你是谁"
        print('documents size: ', len(documents))
        if len(documents) != 0:
            prompt = "\n".join([doc.content for doc in documents])
            prompt = prompt + "请根据以上知识点，回答下面问题：" + query
        else:
            prompt = query
        print('--------- all prompt:----------:\n', prompt)
        return {"prompt": prompt}

def haystack_stream(chunk: StreamingChunk):
    # print(chunk.content, end="", flush=True)
    stream_queue.put_nowait(chunk.content)
    pass

# 初始化 OpenAI Generator 节点
openai_generator = OpenAIGenerator(
    api_base_url="https://open.bigmodel.cn/api/paas/v4/",
    api_key=Secret.from_env_var("CHATGLM_API_KEY"),  # 替换为你的 OpenAI API 密钥
    model="glm-4-flash",
    system_prompt="你作为高中语文老师，按照要求回答问题即可",
    streaming_callback=haystack_stream
)

# 构建管道
pipeline = Pipeline()
pipeline.add_component(name="retriever", instance=retriever)
pipeline.add_component(name="converter", instance=DocumentToPrompt())
pipeline.add_component(name="generator", instance=openai_generator)
# 连接组件
pipeline.connect("retriever.documents", "converter.documents")
pipeline.connect("converter.prompt", "generator.prompt")

@app.get("/hello")
async def hello() -> str:
    return 'helloworld'


def startRag(query, top_k):
    result = pipeline.run(
        data={
            "retriever": {"query": query, "top_k": top_k},  # 传递查询和 top_k 参数
            "converter": {"query": query}
        }
    )
    return result

@app.get("/query-stream")
async def query_stream(query: str, top_k: int = Query(3)):
    print('query:', query)
    print('top_k:', top_k)
    """
    流式接口：通过 StreamingResponse 返回流式结果
    """
    print('rag 流式返回接口')
    async def event_stream():  # define async generation
        while True:
            content = await stream_queue.get()
            # print(f"-------------:{content}", end="", flush=True)
            if content is None:
                break
            yield f"data: {content}\n\n"    # 必须要加 \n\n ，负责会报错：输出都是空
        yield "data: [END]\n\n"  # sent stop signal
        print("Stream ended.")
    async def produce_data():
        await asyncio.to_thread(startRag, query, top_k)
        await stream_queue.put(None)  # 结束信号
    asyncio.create_task(produce_data())
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/query-no-stream")
async def query_stream(query: str, top_k: int = Query(3)):
    print('query:', query)
    print('top_k:', top_k)
    """
    流式接口：通过 StreamingResponse 返回流式结果
    """
    results = startRag(query=query, top_k=top_k)
    answer = '无返回值'
    if results["generator"] != None and results["generator"]["replies"] != None and results["generator"]["replies"][0] != None:
        answer = results["generator"]["replies"][0]
    return {"answer": answer}


'''
运行命令: nohup uvicorn web:app --host 0.0.0.0 --port 8000 --reload &

基础版本：
实现了基本的流式输出和非流失输出
流式输出还存在问题: 异步队列是全局的，当同一时刻发送多请求，会导致回答混乱。
'''
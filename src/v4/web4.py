from fastapi import FastAPI, Query
from haystack import Document
from fastapi.responses import StreamingResponse
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack import Pipeline
from haystack.utils import Secret
from haystack import component
from haystack.dataclasses import StreamingChunk
from typing import Any, Callable, Dict, List, Optional

import asyncio

import os, sys

print("未改变前 sys.path:", sys.path)

# 获取当前文件所在目录的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "..", ".."))

# 将 src 目录加入 sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
    # sys.path.insert(0, project_root)  # 确保 src 在 sys.path 的最前面

# 打印 sys.path 以确认
print("project_root:", project_root)
print("改变后 sys.path:", sys.path)

app = FastAPI()

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

from src.v4.chatglm import ChatGLM
from src.v4.documentToPrompt import DocumentToPrompt
openai_generator = ChatGLM(
    system_prompt="你作为高中语文老师，按照要求回答问题即可"
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

    # 创建独立队列
    stream_queue = asyncio.Queue()

    def haystack_stream(chunk: StreamingChunk):
        print(chunk.content, end="", flush=True)
        stream_queue.put_nowait(chunk.content)
    pass
    
    openai_generator.set_haystack_streaming_callback(haystack_stream)
    """
    流式接口：通过 StreamingResponse 返回流式结果
    """
    print('rag 流式返回接口')
    async def event_stream():  # define async generation
        while True:
            content = await stream_queue.get()
            print(f"-------------:{content}", end="", flush=True)
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
运行命令: nohup uvicorn web4:app --host 0.0.0.0 --port 8004 --reload > nohup4.out 2>&1 &

基础版本3: 实现了基本的流式输出和非流式输出

和上一个版本相比: 将 ChatGLM 和 DocumentToPrompt 简单提取了出来。私有队列还是没有动。v3是最开始迭代版本。
'''
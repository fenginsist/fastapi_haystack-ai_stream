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

from src.v5.rag import startRag # 必须要在上面更新工作目录之后。

app = FastAPI()

@app.get("/hello")
async def hello() -> str:
    return 'helloworld'


@app.get("/query-stream")
async def query_stream(query: str, top_k: int = Query(3)):
    print('query:', query)
    print('top_k:', top_k)

    # 创建独立队列
    stream_queue = asyncio.Queue()
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
        await asyncio.to_thread(startRag, query, top_k, stream_queue)
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
运行命令: nohup uvicorn web5:app --host 0.0.0.0 --port 8005 --reload > nohup5.out 2>&1 &

当前情况: 实现了基本的流式输出和非流式输出

当前问题: 发现老问题并未解决, 当请求A 和 请求B(比请求A晚一些) 同时发起时, 请求B会占用请求A, 请求A的流式输出就戛然而止, 只有请求B在流式输出。而且，请求A的内容会在请求B上输出一小部分。

问题原因: 当两个请求同时到来时, stream_queue 会被后一个请求覆盖, 导致前一个请求的数据也被写入到新的队列中。

和上一个版本相比: 将rag管道的代码抽取成了 rag.py 文件。web文件只有请求的端口。将回调函数整合到了 startRag中。
'''
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



def startRag(query, top_k, stream_queue:Optional[asyncio.Queue]=None):
    if stream_queue is None: # 如果 stream_queue 没有传入，创建一个新的 asyncio.Queue 实例
        stream_queue = asyncio.Queue() 
        pass
    
    def haystack_stream(chunk: StreamingChunk):
        print(chunk.content, end="", flush=True)
        stream_queue.put_nowait(chunk.content)
    pass
    
    openai_generator.set_haystack_streaming_callback(haystack_stream)

    result = pipeline.run(
        data={
            "retriever": {"query": query, "top_k": top_k},  # 传递查询和 top_k 参数
            "converter": {"query": query}
        }
    )
    return result
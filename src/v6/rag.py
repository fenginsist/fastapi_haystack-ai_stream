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

from src.v6.chatglm import ChatGLM
from src.v6.documentToPrompt import DocumentToPrompt


class Rag:
    def __init__(self, 
                 stream_queue: Optional[asyncio.Queue]=None):
        
        # 初始化文档存储 并 写入文档
        self.document_store = InMemoryDocumentStore()
        documents = [Document(content="Haystack is an open-source NLP framework."),
                    Document(content="FastAPI is a modern web framework for Python."),
                    Document(content="Streaming responses can improve real-time user experiences.")
        ]
        self.document_store.write_documents(documents=documents)

        # check 文档存储中的文档
        all_docs = self.document_store.count_documents()
        if all_docs == len(documents):
            print("document size:", all_docs)
            print("All documents were successfully written to the document store.")
        else:
            print("document size:", all_docs)
            print("Not all documents were written to the document store.")
        
        # 初始化检索器 
        self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
        
        self.openai_generator = ChatGLM(system_prompt="你作为高中语文老师，按照要求回答问题即可")

        # 构建管道
        self.pipeline = Pipeline()
        self.pipeline.add_component(name="retriever", instance=self.retriever)
        self.pipeline.add_component(name="converter", instance=DocumentToPrompt())
        self.pipeline.add_component(name="generator", instance=self.openai_generator)
        # 连接组件
        self.pipeline.connect("retriever.documents", "converter.documents")
        self.pipeline.connect("converter.prompt", "generator.prompt")
        pass
    


    def startRag(self, query: str, top_k: int, stream_queue:Optional[asyncio.Queue]=None):
        # self.stream_queue = stream_queue
        ChatGLM.stream_queue = stream_queue
        result = self.pipeline.run(
            data={
                "retriever": {"query": query, "top_k": top_k},  # 传递查询和 top_k 参数
                "converter": {"query": query}
            }
        )
        return result
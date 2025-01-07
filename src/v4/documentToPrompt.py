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

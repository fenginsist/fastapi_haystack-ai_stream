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

import contextlib

class ChatGLM(OpenAIGenerator):
    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("CHATGLM_API_KEY"),
        model: str = "glm-4-flash",
        haystack_streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = "https://open.bigmodel.cn/api/paas/v4/",
        organization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            streaming_callback=haystack_streaming_callback,
            api_base_url=api_base_url,
            organization=organization,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )
        pass

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        output = OpenAIGenerator.run(
            self=self,
            prompt=prompt,
            streaming_callback=streaming_callback
        )
        print('ChatGLM output:', output)
        return {
            "replies": output['replies'],
            "meta": output['meta'],
        }

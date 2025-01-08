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
        stream_queue: Optional[asyncio.Queue] = None,
    ):
        # self.dynamic_callback = haystack_streaming_callback
        self.stream_queue = stream_queue
        super().__init__(
            api_key=api_key,
            model=model,
            streaming_callback=self.haystack_stream,
            api_base_url=api_base_url,
            organization=organization,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )
        pass
    def set_haystack_streaming_callback(self, callback: Callable[[StreamingChunk], None]):
        """
        动态设置流式回调函数。
        """
        self.dynamic_callback = callback

    def dynamic_streaming_callback(self, chunk: StreamingChunk):
        """
        内部流式回调函数，调用动态设置的回调。
        """
        if self.dynamic_callback:
            self.dynamic_callback(chunk)
            
    def haystack_stream(chunk: StreamingChunk):
        print(chunk.content, end="", flush=True)
        self.stream_queue.put_nowait(chunk.content)
        pass

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
    ):
        output = OpenAIGenerator.run(
            self=self,
            prompt=prompt
        )
        print('ChatGLM output:', output)
        return {
            "replies": output['replies'],
            "meta": output['meta'],
        }

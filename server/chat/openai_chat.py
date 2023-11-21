from fastapi.responses import StreamingResponse
from typing import List, Optional,Literal
import openai
from configs import LLM_MODELS, logger, log_verbose
from server.utils import get_model_worker_config, fschat_openai_api_address
from pydantic import BaseModel,Field
from server.db.repository import add_chat_history_to_db, update_chat_history
import json


class OpenAiMessage(BaseModel):
    role: str = "user"
    content: str = "hello"


class OpenAiChatMsgIn(BaseModel):
    model: str = LLM_MODELS[0]
    messages: List[OpenAiMessage]
    temperature: float = 0.7
    n: int = 1
    max_tokens: Optional[int] = None
    stop: List[str] = []
    stream: bool = False
    presence_penalty: int = 0
    frequency_penalty: int = 0

###增加内容
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)

    model_config = {
        "json_schema_extra": {"examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]}
    }


async def openai_chat(msg: ChatCompletionRequest):
    config = get_model_worker_config(msg.model)
    openai.api_key = config.get("api_key", "EMPTY")
    print(f"{openai.api_key=}")
    openai.api_base = config.get("api_base_url", fschat_openai_api_address())
    print(fschat_openai_api_address())
    print(f"{openai.api_base=}")
    print(msg)

    async def get_response(msg):
        data = msg.dict()
        ###这里openai传参方式，要一个个传，不能直接**data
        try:
            response = await openai.ChatCompletion.acreate(model=data['model'],
                                        messages=data['messages'],
                                        temperature=data['temperature'],
                                        top_p=data['top_p'],
                                        stream=data['stream'],
                                        max_tokens=data['max_tokens'])
            chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=data['messages'][0]['content'])
            if msg.stream:
                async for data in response:
                    if choices := data.choices:
                        if chunk := choices[0].get("delta", {}).get("content"):
                            print(chunk, end="", flush=True)
                            yield json.dumps(
                                {"text": chunk, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
            else:
                if response.choices:
                    answer = response.choices[0].message.content
                    print(answer)
                    yield json.dumps(
                                {"text": answer, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
        except Exception as e:
            msg = f"获取ChatCompletion时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)

    return StreamingResponse(
        get_response(msg),
        media_type='text/event-stream',
    )

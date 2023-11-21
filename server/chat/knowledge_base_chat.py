from fastapi.responses import StreamingResponse
from typing import List, Optional,Literal,Dict
import openai
from configs import LLM_MODELS, logger, log_verbose,VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE
from server.utils import get_model_worker_config, fschat_openai_api_address,BaseResponse,get_prompt_template
from pydantic import BaseModel,Field
from server.db.repository import add_chat_history_to_db, update_chat_history
import json
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from fastapi import Body, Request
from server.knowledge_base.kb_doc_api import search_docs
from langchain.prompts.chat import ChatPromptTemplate


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


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                            knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                            top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                            score_threshold: float = Body(SCORE_THRESHOLD, description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右", ge=0, le=2),
                            history: List[Dict] = Body([],
                                                      description="历史对话",
                                                      examples=[[
                                                          {"role": "user",
                                                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                          {"role": "assistant",
                                                           "content": "虎头虎脑"}]]
                                                      ),
                            stream: bool = Body(False, description="流式输出"),
                            model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                            temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                            max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                            prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                            request: Request = None,
                        ):

    print('--------------------------------')
    config = get_model_worker_config(model_name)
    openai.api_key = config.get("api_key", "EMPTY")
    print(f"{openai.api_key=}")
    openai.api_base = config.get("api_base_url", fschat_openai_api_address())
    print(fschat_openai_api_address())
    print(f"{openai.api_base=}")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    
    docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
    context = "\n".join([doc.page_content for doc in docs])
    prompts = '\n'.join([i['content'] for i in history])
    msg = {
        'model':model_name,
        'messages': [{'role':'user','content':f'<指令>请根据用户的问题，结合给出的上下文，进行简洁明了的回答。你的回答不需要重复用户的问题及给出的上下文。</指令>\n<问题>{query}</问题>\n<上下文>{context}<上下文>'}],
        'temperature':temperature,
        'top_p':0.95,
        'stream':False,
        'max_tokens':max_tokens or 2048
    }
    async def get_response(msg):
        data = msg
        print('&'*60)
        ###这里openai传参方式，要一个个传，不能直接**data
        try:
            response = await openai.ChatCompletion.acreate(model=data['model'],
                                        messages=data['messages'],
                                        temperature=data['temperature'],
                                        top_p=data['top_p'],
                                        stream=data['stream'],
                                        max_tokens=data['max_tokens'])
            chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=data['messages'][0]['content'])
            if stream:
                async for data in response:
                    if choices := data.choices:
                        if chunk := choices[0].get("delta", {}).get("content"):
                            print(chunk, end="", flush=True)
                            yield json.dumps(
                                {"answer": chunk, "docs": []},
                                ensure_ascii=False)
            else:
                if response.choices:
                    answer = response.choices[0].message.content
                    print(answer)
                    yield json.dumps(
                                {"answer": answer, "docs": []},
                                ensure_ascii=False)
        except Exception as e:
            msg = f"获取ChatCompletion时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
    
    
    return StreamingResponse(
        get_response(msg),
        media_type='text/event-stream',
    )

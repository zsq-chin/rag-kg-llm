import os
import json
import asyncio
import traceback
import uuid
import time
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src import executor, config, retriever
from src.core import HistoryManager
from src.agents import agent_manager
from src.models import select_model
from src.utils.logging_config import logger
from src.agents.tools_factory import get_all_tools
from server.routers.auth_router import get_admin_user
from server.utils.auth_middleware import get_required_user, get_db
from server.models.user_model import User
from server.models.thread_model import Thread
from server.models.chat_model import ChatRecord, ExamPapersRecord, GuideRecord, ItemRecord, WriterRecord

chat = APIRouter(prefix="/chat")

@chat.get("/default_agent")
async def get_default_agent(current_user: User = Depends(get_required_user)):
    """获取默认智能体ID（需要登录）"""
    try:
        default_agent_id = config.default_agent_id
        # 如果没有设置默认智能体，尝试获取第一个可用的智能体
        if not default_agent_id:
            agents = await agent_manager.get_agents_info()
            if agents:
                default_agent_id = agents[0].get("name", "")

        return {"default_agent_id": default_agent_id}
    except Exception as e:
        logger.error(f"获取默认智能体出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取默认智能体出错: {str(e)}")

@chat.post("/set_default_agent")
async def set_default_agent(agent_id: str = Body(..., embed=True), current_user = Depends(get_admin_user)):
    """设置默认智能体ID (仅管理员)"""
    try:
        # 验证智能体是否存在
        agents = await agent_manager.get_agents_info()
        agent_ids = [agent.get("name", "") for agent in agents]

        if agent_id not in agent_ids:
            raise HTTPException(status_code=404, detail=f"智能体 {agent_id} 不存在")

        # 设置默认智能体ID
        config.default_agent_id = agent_id
        # 保存配置
        config.save()

        return {"success": True, "default_agent_id": agent_id}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"设置默认智能体出错: {e}")
        raise HTTPException(status_code=500, detail=f"设置默认智能体出错: {str(e)}")

@chat.get("/")
async def chat_get(current_user: User = Depends(get_required_user)):
    """聊天服务健康检查（需要登录）"""
    return "Chat Get!"

@chat.post("/")
async def chat_post(
        #  ...代表必填，None代表可选 
        query: str = Body(...),
        meta: dict = Body(None),
        history: list[dict] | None = Body(None),
        thread_id: str | None = Body(None),
        current_user: User = Depends(get_required_user)):
    """处理聊天请求的主要端点（需要登录）"""

    model = select_model()
    meta["server_model_name"] = model.model_name
    history_manager = HistoryManager(history, system_prompt=meta.get("system_prompt"))
    logger.debug(f"Received query: {query} with meta: {meta}")

    # 构造一条 JSON 格式的数据块（chunk），并编码成 字节串，末尾再加上一个换行符 b"\n"用于 流式响应
    # 形如：
    #     {
    #     "response": " ask",
    #     "meta": {
    #         "use_graph": false,
    #         "use_web": false,
    #         "graph_name": "neo4j",
    #         "selectedKB": null,
    #         "summary_title": false,
    #         "history_round": 20,
    #         "fontSize": "default",
    #         "wideScreen": false,
    #         "server_model_name": "Qwen/Qwen2.5-72B-Instruct"
    #     },
    #     "status": "loading"
    # }
    def make_chunk(content=None, **kwargs):
        return json.dumps({
            "response": content,
            "meta": meta,
            **kwargs
        }, ensure_ascii=False).encode('utf-8') + b"\n"

    def need_retrieve(meta):
        return meta.get("use_web") or meta.get("use_graph") or meta.get("db_id")

    def generate_response():
        modified_query = query
        refs = None

        # 处理知识库检索
        if meta and need_retrieve(meta):
            chunk = make_chunk(status="searching")
            yield chunk

            try:
                modified_query, refs = retriever(modified_query, history_manager.messages, meta)
            except Exception as e:
                logger.error(f"Retriever error: {e}, {traceback.format_exc()}")
                yield make_chunk(message=f"Retriever error: {e}", status="error")
                return

            yield make_chunk(status="generating")

        messages = history_manager.get_history_with_msg(modified_query, max_rounds=meta.get('history_round'))
        history_manager.add_user(query)  # 注意这里使用原始查询

        content = ""
        reasoning_content = ""
        try:
            for delta in model.predict(messages, stream=True):
                # 推理模型才会有reasoning_content属性
                if not delta.content and hasattr(delta, 'reasoning_content'):
                    reasoning_content += delta.reasoning_content or ""
                    chunk = make_chunk(reasoning_content=reasoning_content, status="reasoning")
                    yield chunk
                    continue

                # 文心一言
                if hasattr(delta, 'is_full') and delta.is_full:
                    content = delta.content
                else:
                    content += delta.content or ""

                chunk = make_chunk(content=delta.content, status="loading")
                yield chunk

            logger.debug(f"Final response: {content}")
            logger.debug(f"Final reasoning response: {reasoning_content}")
            yield make_chunk(status="finished",
                            history=history_manager.update_ai(content),
                            refs=refs)
        except Exception as e:
            logger.error(f"Model error: {e}, {traceback.format_exc()}")
            yield make_chunk(message=f"Model error: {e}", status="error")
            return

    return StreamingResponse(generate_response(), media_type='application/json')

@chat.post("/call")
async def call(query: str = Body(...), meta: dict = Body(None), current_user: User = Depends(get_required_user)):
    """调用模型进行简单问答（需要登录）"""
    meta = meta or {}
    model = select_model(model_provider=meta.get("model_provider"), model_name=meta.get("model_name"))
    async def predict_async(query):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, model.predict, query)

    response = await predict_async(query)
    logger.debug({"query": query, "response": response.content})

    return {"response": response.content}

@chat.get("/agent")
async def get_agent(current_user: User = Depends(get_required_user)):
    """获取所有可用智能体（需要登录）"""
    agents = await agent_manager.get_agents_info()
    # logger.debug(f"agents: {agents}")
    return {"agents": agents}

@chat.post("/agent/{agent_name}")
async def chat_agent(agent_name: str,
               query: str = Body(...),
               config: dict = Body({}),
               meta: dict = Body({}),
               current_user: User = Depends(get_required_user)):
    """使用特定智能体进行对话（需要登录）"""

    meta.update({
        "query": query,
        "agent_name": agent_name,
        "server_model_name": config.get("model", agent_name),
        "thread_id": config.get("thread_id"),
        "user_id": current_user.id
    })

    # 将meta和thread_id整合到config中
    def make_chunk(content=None, **kwargs):

        return json.dumps({
            "request_id": meta.get("request_id"),
            "response": content,
            **kwargs
        }, ensure_ascii=False).encode('utf-8') + b"\n"

    async def stream_messages():

        # 代表服务端已经收到了请求
        yield make_chunk(status="init", meta=meta, msg=HumanMessage(content=query).model_dump())

        try:
            agent = agent_manager.get_agent(agent_name)
        except Exception as e:
            logger.error(f"Error getting agent {agent_name}: {e}, {traceback.format_exc()}")
            yield make_chunk(message=f"Error getting agent {agent_name}: {e}", status="error")
            return

        messages = [{"role": "user", "content": query}]

        # 构造运行时配置，如果没有thread_id则生成一个
        config["user_id"] = current_user.id
        if "thread_id" not in config or not config["thread_id"]:
            config["thread_id"] = str(uuid.uuid4())
            logger.debug(f"没有thread_id，生成一个: {config['thread_id']=}")

        runnable_config = {"configurable": {**config}}

        try:
            async for msg, metadata in agent.stream_messages(messages, config_schema=runnable_config):
                # logger.debug(f"msg: {msg.model_dump()}, metadata: {metadata}")
                if isinstance(msg, AIMessageChunk):
                    yield make_chunk(content=msg.content,
                                    msg=msg.model_dump(),
                                    metadata=metadata,
                                    status="loading")
                else:
                    yield make_chunk(msg=msg.model_dump(),
                                    metadata=metadata,
                                    status="loading")

            yield make_chunk(status="finished", meta=meta)
        except Exception as e:
            logger.error(f"Error streaming messages: {e}, {traceback.format_exc()}")
            yield make_chunk(message=f"Error streaming messages: {e}", status="error")

    return StreamingResponse(stream_messages(), media_type='application/json')

@chat.get("/models")
async def get_chat_models(model_provider: str, current_user: User = Depends(get_admin_user)):
    """获取指定模型提供商的模型列表（需要登录）"""
    model = select_model(model_provider=model_provider)
    return {"models": model.get_models()}

@chat.post("/models/update")
async def update_chat_models(model_provider: str, model_names: list[str], current_user = Depends(get_admin_user)):
    """更新指定模型提供商的模型列表 (仅管理员)"""
    config.model_names[model_provider]["models"] = model_names
    config._save_models_to_file()
    return {"models": config.model_names[model_provider]["models"]}

@chat.get("/tools")
async def get_tools(current_user: User = Depends(get_admin_user)):
    """获取所有可用工具（需要登录）"""
    return {"tools": list(get_all_tools().keys())}

@chat.post("/agent/{agent_name}/config")
async def save_agent_config(
    agent_name: str,
    config: dict = Body(...),
    current_user: User = Depends(get_admin_user)
):
    """保存智能体配置到YAML文件（需要管理员权限）"""
    try:
        # 获取Agent实例和配置类
        agent = agent_manager.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"智能体 {agent_name} 不存在")

        # 使用配置类的save_to_file方法保存配置
        config_cls = agent.config_schema
        result = config_cls.save_to_file(config, agent_name)

        if result:
            return {"success": True, "message": f"智能体 {agent_name} 配置已保存"}
        else:
            raise HTTPException(status_code=500, detail="保存智能体配置失败")

    except Exception as e:
        logger.error(f"保存智能体配置出错: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"保存智能体配置出错: {str(e)}")

@chat.get("/agent/{agent_name}/history")
async def get_agent_history(
    agent_name: str,
    thread_id: str,
    current_user: User = Depends(get_required_user)
):
    """获取智能体历史消息（需要登录）"""
    try:
        # 获取Agent实例和配置类
        agent = agent_manager.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail=f"智能体 {agent_name} 不存在")

        # 获取历史消息
        history = await agent.get_history(user_id=current_user.id, thread_id=thread_id)
        return {"history": history}

    except Exception as e:
        logger.error(f"获取智能体历史消息出错: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"获取智能体历史消息出错: {str(e)}")

@chat.get("/agent/{agent_name}/config")
async def get_agent_config(
    agent_name: str,
    current_user: User = Depends(get_required_user)
):
    """从YAML文件加载智能体配置（需要登录）"""
    try:
        # 检查智能体是否存在
        if not (agent := agent_manager.get_agent(agent_name)):
            raise HTTPException(status_code=404, detail=f"智能体 {agent_name} 不存在")

        config = agent.config_schema.from_runnable_config(config={}, agent_name=agent_name)
        return {"success": True, "config": config}

    except Exception as e:
        logger.error(f"加载智能体配置出错: {e}, {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"加载智能体配置出错: {str(e)}")

# ==================== 线程管理 API ====================

class ThreadCreate(BaseModel):
    title: str | None = None
    agent_id: str
    description: str | None = None
    metadata: dict | None = None


class ThreadResponse(BaseModel):
    id: str
    user_id: str
    agent_id: str
    title: str | None = None
    description: str | None = None
    create_at: str
    update_at: str


@chat.post("/thread", response_model=ThreadResponse)
async def create_thread(
    thread: ThreadCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """创建新对话线程"""
    thread_id = str(uuid.uuid4())

    new_thread = Thread(
        id=thread_id,
        user_id=current_user.id,
        agent_id=thread.agent_id,
        title=thread.title or "新对话",
        description=thread.description,
    )

    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)

    return {
        "id": new_thread.id,
        "user_id": new_thread.user_id,
        "agent_id": new_thread.agent_id,
        "title": new_thread.title,
        "description": new_thread.description,
        "create_at": new_thread.create_at.isoformat(),
        "update_at": new_thread.update_at.isoformat(),
    }


@chat.get("/threads", response_model=list[ThreadResponse])
async def list_threads(
    agent_id: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """获取用户的所有对话线程"""
    query = db.query(Thread).filter(
        Thread.user_id == current_user.id,
        Thread.status == 1
    )

    if agent_id:
        query = query.filter(Thread.agent_id == agent_id)

    threads = query.order_by(Thread.update_at.desc()).all()

    return [
        {
            "id": thread.id,
            "user_id": thread.user_id,
            "agent_id": thread.agent_id,
            "title": thread.title,
            "description": thread.description,
            "create_at": thread.create_at.isoformat(),
            "update_at": thread.update_at.isoformat(),
        }
        for thread in threads
    ]


@chat.delete("/thread/{thread_id}")
async def delete_thread(
    thread_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """删除对话线程"""
    thread = db.query(Thread).filter(
        Thread.id == thread_id,
        Thread.user_id == current_user.id
    ).first()

    if not thread:
        raise HTTPException(status_code=404, detail="对话线程不存在")

    # 软删除
    thread.status = 0
    db.commit()

    return {"message": "删除成功"}


class ThreadUpdate(BaseModel):
    title: str | None = None
    description: str | None = None


@chat.put("/thread/{thread_id}", response_model=ThreadResponse)
async def update_thread(
    thread_id: str,
    thread_update: ThreadUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    """更新对话线程信息"""
    thread = db.query(Thread).filter(
        Thread.id == thread_id,
        Thread.user_id == current_user.id,
        Thread.status == 1
    ).first()

    if not thread:
        raise HTTPException(status_code=404, detail="对话线程不存在")

    if thread_update.title is not None:
        thread.title = thread_update.title

    if thread_update.description is not None:
        thread.description = thread_update.description

    db.commit()
    db.refresh(thread)

    return {
        "id": thread.id,
        "user_id": thread.user_id,
        "agent_id": thread.agent_id,
        "title": thread.title,
        "description": thread.description,
        "create_at": thread.create_at.isoformat(),
        "update_at": thread.update_at.isoformat(),
    }

# ==================== 聊天记录相关 API ====================

from datetime import datetime
from pytz import timezone
import json
beijing = timezone('Asia/Shanghai')
from zoneinfo import ZoneInfo

@chat.get("/records", response_model=list)
async def get_chat_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    records = db.query(ChatRecord).filter(ChatRecord.user_id == current_user.id).order_by(ChatRecord.updatetime.desc()).all()
    return [
        {
            "id": r.id,
            "content": json.loads(r.content),
            "updatetime": r.updatetime.isoformat()
        }
        for r in records
    ]

@chat.post("/records")
async def save_chat_record(
    record: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 取出新记录的 id
    new_id = str(record.get("id"))

    if not new_id:
        return {"success": False, "msg": "记录中缺少 id 字段"}

    # 删除 id 相同的旧记录（通过 conv_id）
    db.query(ChatRecord).filter(
        ChatRecord.user_id == current_user.id,
        ChatRecord.conv_id == new_id
    ).delete()

    # 添加新记录
    db.add(ChatRecord(
        content=json.dumps(record, ensure_ascii=False),
        conv_id=new_id,
        user_id=current_user.id,
        updatetime=datetime.now(ZoneInfo("Asia/Shanghai"))
    ))

    db.commit()
    return {"success": True}

@chat.delete("/records/{record_id}")
async def delete_chat_record(
    record_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 查找并删除当前用户指定 conv_id 的记录
    result = db.query(ChatRecord).filter(
        ChatRecord.user_id == current_user.id,
        ChatRecord.conv_id == record_id
    ).delete(synchronize_session=False)

    if result:
        db.commit()
        return {"success": True, "msg": f"记录 {record_id} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 {record_id} 的记录")


# ==================== 引导记录相关 API ====================

from datetime import datetime
from pytz import timezone
import json
from zoneinfo import ZoneInfo

@chat.get("/guide", response_model=list)
async def get_guide_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    records = db.query(GuideRecord).filter(GuideRecord.user_id == current_user.id).order_by(GuideRecord.updatetime.desc()).all()
    return [
        {
            "id": r.id,
            "content": json.loads(r.content),
            "updatetime": r.updatetime.isoformat()
        }
        for r in records
    ]

@chat.post("/guide")
async def save_chat_record(
    record: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 取出新记录的 id
    new_id = str(record.get("id"))

    if not new_id:
        return {"success": False, "msg": "记录中缺少 id 字段"}

    # 删除 id 相同的旧记录（通过 conv_id）
    db.query(GuideRecord).filter(
        GuideRecord.user_id == current_user.id,
        GuideRecord.guide_id == new_id
    ).delete()

    # 添加新记录
    db.add(GuideRecord(
        content=json.dumps(record, ensure_ascii=False),
        guide_id=new_id,
        user_id=current_user.id,
        updatetime=datetime.now(ZoneInfo("Asia/Shanghai"))
    ))

    db.commit()
    return {"success": True}

@chat.delete("/guide/{guide_id}")
async def delete_chat_record(
    guide_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 查找并删除当前用户指定 guide_id 的记录
    result = db.query(GuideRecord).filter(
        GuideRecord.user_id == current_user.id,
        GuideRecord.guide_id == guide_id
    ).delete(synchronize_session=False)

    if result:
        db.commit()
        return {"success": True, "msg": f"记录 {guide_id} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 {guide_id} 的记录")
    

# ==================== 写作达人聊天记录相关 API ====================

from datetime import datetime
from pytz import timezone
import json
beijing = timezone('Asia/Shanghai')
from zoneinfo import ZoneInfo

@chat.get("/writer", response_model=list)
async def get_writer_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    records = db.query(WriterRecord).filter(WriterRecord.user_id == current_user.id).order_by(WriterRecord.updatetime.desc()).all()
    return [
        {
            "id": r.id,
            "content": json.loads(r.content),
            "updatetime": r.updatetime.isoformat()
        }
        for r in records
    ]

@chat.post("/writer")
async def save_writer_record(
    record: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 取出新记录的 id
    new_id = str(record.get("id"))

    if not new_id:
        return {"success": False, "msg": "记录中缺少 id 字段"}

    # 删除 id 相同的旧记录（通过 conv_id）
    db.query(WriterRecord).filter(
        WriterRecord.user_id == current_user.id,
        WriterRecord.conv_id == new_id
    ).delete()

    # 添加新记录
    db.add(WriterRecord(
        content=json.dumps(record, ensure_ascii=False),
        conv_id=new_id,
        user_id=current_user.id,
        updatetime=datetime.now(ZoneInfo("Asia/Shanghai"))
    ))

    db.commit()
    return {"success": True}

@chat.delete("/writer/{writer_id}")
async def delete_writer_record(
    writer_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    logger.info(f"Deleting writer record with ID: {writer_id} for user {current_user.id}")
    # 查找并删除当前用户指定 conv_id 的记录
    result = db.query(WriterRecord).filter(
        WriterRecord.user_id == current_user.id,
        WriterRecord.conv_id == writer_id
    ).delete(synchronize_session=False)

    if result:
        db.commit()
        return {"success": True, "msg": f"记录 {writer_id} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 {writer_id} 的记录")
    

# ==================== 题目生成记录相关 API ====================

from datetime import datetime
from pytz import timezone
import json
from zoneinfo import ZoneInfo

@chat.get("/item", response_model=list)
async def get_item_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    records = db.query(ItemRecord).filter(ItemRecord.user_id == current_user.id).order_by(ItemRecord.createdtime.desc()).all()
    return [
        {
            "id": r.id,
            "content": json.loads(r.content),
            "createdtime": r.createdtime.isoformat()
        }
        for r in records
    ]

@chat.post("/item")
async def save_item_record(
    record: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 取出新记录的 id
    new_id = str(record.get("id"))

    if not new_id:
        return {"success": False, "msg": "记录中缺少 id 字段"}

    # 删除 id 相同的旧记录（通过 item_id）
    db.query(ItemRecord).filter(
        ItemRecord.user_id == current_user.id,
        ItemRecord.item_id == new_id
    ).delete()

    # 添加新记录
    db.add(ItemRecord(
        content=json.dumps(record, ensure_ascii=False),
        item_id=new_id,
        user_id=current_user.id,
        structured_content=record.get("structured_content") or '',
        createdtime=datetime.now(ZoneInfo("Asia/Shanghai"))
    ))

    db.commit()
    return {"success": True}

@chat.delete("/item/{item_id}")
async def delete_item_record(
    item_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 查找并删除当前用户指定 item_id 的记录
    result = db.query(ItemRecord).filter(
        ItemRecord.user_id == current_user.id,
        ItemRecord.item_id == item_id
    ).delete(synchronize_session=False)

    if result:
        db.commit()
        return {"success": True, "msg": f"记录 {item_id} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 {item_id} 的记录")


# ==================== 空白试卷记录相关 API ====================

from datetime import datetime
from pytz import timezone
import json
from zoneinfo import ZoneInfo

@chat.get("/exam", response_model=list)
async def get_exam_records(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    records = db.query(ExamPapersRecord).filter(ExamPapersRecord.user_id == current_user.id).order_by(ExamPapersRecord.createdtime.desc()).all()
    return [
        {
            "id": r.id,
            "content": json.loads(r.content),
            "submission_content": json.loads(r.submission_content) if r.submission_content and r.submission_content.strip() else None,
            "createdtime": r.createdtime.isoformat()
        }
        for r in records
    ]

@chat.post("/exam")
async def save_exam_record(
    record: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 取出新记录的 id
    new_id = str(record.get("id"))

    if not new_id:
        return {"success": False, "msg": "记录中缺少 id 字段"}

    # 删除 id 相同的旧记录（通过 item_id）
    db.query(ExamPapersRecord).filter(
        ExamPapersRecord.user_id == current_user.id,
        ExamPapersRecord.exam_paper_id == new_id
    ).delete()

    # 添加新记录
    db.add(ExamPapersRecord(
        content=json.dumps(record, ensure_ascii=False),
        exam_paper_id=new_id,
        user_id=current_user.id,
        createdtime=datetime.now(ZoneInfo("Asia/Shanghai"))
    ))

    db.commit()
    return {"success": True}

from fastapi import Body, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

@chat.post("/exam_sub")
async def save_exam_sub_record(
    record: dict = Body(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    new_id = str(record.get("id"))

    if not new_id:
        return {"success": False, "msg": "记录中缺少 id 字段"}

    temp_record = db.query(ExamPapersRecord).filter(
        ExamPapersRecord.user_id == current_user.id,
        ExamPapersRecord.exam_paper_id == new_id
    ).first()  # 注意要取 first，否则 temp_record 是 Query 对象

    if not temp_record:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 {new_id} 的记录")

    try:
        temp_record.submission_content = json.dumps(record, ensure_ascii=False)
        db.commit()
        db.refresh(temp_record)  # 刷新对象，确保最新数据
        return {"success": True, "record": record}
    except SQLAlchemyError as e:
        db.rollback()  # 回滚事务
        return {"success": False, "msg": "数据库更新失败", "error": str(e)}


@chat.delete("/exam/{exam_paper_id}")
async def delete_exam_record(
    exam_paper_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_required_user)
):
    # 查找并删除当前用户指定 item_id 的记录
    result = db.query(ExamPapersRecord).filter(
        ExamPapersRecord.user_id == current_user.id,
        ExamPapersRecord.exam_paper_id == exam_paper_id
    ).delete(synchronize_session=False)

    if result:
        db.commit()
        return {"success": True, "msg": f"记录 {exam_paper_id} 删除成功"}
    else:
        raise HTTPException(status_code=404, detail=f"未找到 id 为 {exam_paper_id} 的记录")
    
## ============ 第15章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 15: Inter-Agent Communication / A2A）============
# 1. A2A(智能体间通信) = Google 的开放标准，让"不同框架(LangGraph/CrewAI/ADK)搭的 Agent"能互相发现、委派任务、交换信息。
# 2. 三大角色：用户 / A2A 客户端(代表用户发请求的 Agent) / A2A 服务器(远程 Agent，以"不透明"方式干活，客户端不管它内部)。
# 3. 核心概念：① AgentCard(JSON 数字身份：名字/端点/能力/skills/鉴权) ② 发现(从 /.well-known/agent.json 或注册表找到卡片)
#    ③ 通信=tasks 围绕异步状态(submitted/working/completed)流转，消息由 parts 组成，产物叫 artifact，用 server 生成的 contextId 串联多轮。
# 4. 交互机制：同步请求/响应、异步轮询、流式(SSE)、推送通知(webhook)；走 HTTP + JSON-RPC 2.0；模态无关(文本/音视频)。
# 5. A2A vs MCP：MCP 管"Agent 怎么连外部工具/数据"，A2A 管"Agent 之间怎么发现与协作"——互补不替代。
# 6. 安全：mTLS、审计日志、AgentCard 声明鉴权、OAuth2/API Key 走 Header。

## ============ 旧代码（Google ADK + A2A Starlette 服务，已注释，仅供对比）============
# 原书用 a2a-samples：定义 AgentSkill + AgentCard，包成 ADKAgentExecutor，挂到 A2AStarletteApplication 用 uvicorn 起 HTTP 服务。
# 优点是"真·跨进程/跨框架互操作"(标准 A2A 协议)；缺点是依赖 google-a2a SDK + Google Calendar 凭证，demo 门槛高。
# from a2a.types import AgentCard, AgentSkill, AgentCapabilities
# agent_card = AgentCard(name="Calendar Agent", url="http://host:port/", skills=[skill], capabilities=AgentCapabilities(streaming=True))
# a2a_app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)
# uvicorn.run(a2a_app.build(), host=host, port=port)

## ============ 新代码（DeepSeek + 进程内迷你 A2A 总线，当前生效）============
# 设计取舍：完整 A2A 要 a2a SDK + 真实 HTTP 端点。为让你"开箱即跑"，这里用【进程内的迷你 A2A 总线】复刻
# 核心概念：AgentCard(发现) + sendTask(异步任务状态) + Message/Artifact(通信载体) + 客户端/服务器角色分离 +
# 跨 Agent 委派。结构与原书一一对应，只是把网络层换成了内存调用。
import os
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

if not os.getenv("LEARN_AI_KEY"):
    raise ValueError("在 .env 文件中未找到 LEARN_AI_KEY。请添加它。")
try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0,
        api_key=os.getenv("LEARN_AI_KEY"),
        base_url="https://api.deepseek.com",
    )
    print(f"语言模型已初始化: {getattr(llm, 'model', 'deepseek-v4-flash')}")
except Exception as e:
    print(f"初始化语言模型时出错: {e}")
    llm = None


## --- 1. 迷你 A2A 总线：发现 + 任务状态机 + 消息/产物 ---
class A2ABus:
    """复刻 A2A 的'注册中心 + 任务流转'。真实 A2A 这里是 HTTP/JSON-RPC，这里用内存 dict。"""
    def __init__(self):
        self._servers = {}   # name -> (agent_card, handler)

    def register_server(self, card: dict, handler):
        """服务器登记 AgentCard（对应 A2A 的 Agent 发现）。"""
        self._servers[card["name"]] = (card, handler)
        print(f"  📇 服务器已注册 AgentCard：{card['name']} | skills={[s['id'] for s in card['skills']]}")

    def discover(self, name: str) -> dict:
        """客户端'发现'服务器能力（读 AgentCard）。"""
        return self._servers[name][0]

    def send_task(self, client: str, server: str, skill_id: str, payload: str) -> dict:
        """客户端发任务给服务器（对应 A2A 的 tasks/send）。返回带状态的 task 结果。"""
        if server not in self._servers:
            return {"state": "failed", "artifact": f"未知服务器 {server}"}
        card, handler = self._servers[server]
        # 校验客户端请求的 skill 服务器是否声明支持（AgentCard 能力校验）
        if not any(s["id"] == skill_id for s in card["skills"]):
            return {"state": "failed", "artifact": f"{server} 未声明 skill: {skill_id}"}
        task_id = "task-" + uuid.uuid4().hex[:8]
        context_id = "ctx-" + uuid.uuid4().hex[:8]
        print(f"  📨 [{client}] --sendTask--> [{server}] skill={skill_id} task={task_id} ctx={context_id}")
        # 服务器以"不透明"方式处理：客户端只看到 message(parts) 进、artifact 出
        message = {"role": "user", "parts": [{"type": "text", "text": payload}]}
        state, artifact = "working", None
        artifact = handler(skill_id, message)   # 服务器执行
        state = "completed"
        print(f"  📦 [{server}] --artifact--> [{client}] state={state}")
        return {"task_id": task_id, "context_id": context_id, "state": state, "artifact": artifact}


BUS = A2ABus()


## --- 2. 服务器 Agent：天气（提供 get_weather skill）---
WEATHER_DB = {"北京": "晴 22°C", "上海": "多云 25°C", "广州": "雷阵雨 28°C"}
def weather_server_handler(skill_id, message):
    city = message["parts"][0]["text"]
    return f"{city} 天气：{WEATHER_DB.get(city, '暂无数据')}"

BUS.register_server({
    "name": "WeatherBot",
    "description": "提供天气查询",
    "url": "inproc://weatherbot",
    "capabilities": {"streaming": False},
    "skills": [{"id": "get_weather", "name": "Get Weather", "description": "查询城市天气"}],
}, weather_server_handler)


## --- 3. 客户端 Agent：研究员（发现 WeatherBot 并委派任务，最后用 DeepSeek 综合）---
def researcher_agent(user_query: str) -> str:
    """演示 A2A 客户端角色：发现 → 委派 → 综合。"""
    # (a) 发现：读 WeatherBot 的 AgentCard，知道它能 get_weather
    card = BUS.discover("WeatherBot")
    print(f">>> 研究员发现服务器能力：{[s['id'] for s in card['skills']]}")

    # (b) 用 DeepSeek 从问题里抽要查的城市（客户端决定委派什么）
    city = llm.invoke([SystemMessage(content="从问题抽城市名，只回一个词，没有回 NONE。"),
                      HumanMessage(content=user_query)]).content.strip()
    if city.upper() == "NONE":
        return "问题中没有可委派给 WeatherBot 的城市。"

    # (c) 委派：客户端发 sendTask 给服务器（异步任务语义）
    result = BUS.send_task(client="ResearcherAgent", server="WeatherBot",
                           skill_id="get_weather", payload=city)
    if result["state"] != "completed":
        return f"委派失败：{result['artifact']}"

    # (d) 综合：把服务器返回的 artifact 交给 DeepSeek 组织成最终答复
    if llm:
        return llm.invoke([SystemMessage(content="你是研究助手，基于工具返回写一句中文结论。"),
                           HumanMessage(content=f"用户问题：{user_query}\n天气数据：{result['artifact']}")]).content
    return result["artifact"]


if __name__ == "__main__":
    q = "帮我查一下北京和上海今天的天气，并总结一下"
    print(">>> 用户查询：", q)
    print("\n>>> 研究员最终答复：", researcher_agent(q))

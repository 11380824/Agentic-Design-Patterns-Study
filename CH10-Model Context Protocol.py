## ============ 第10章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 10: Model Context Protocol）============
# 1. MCP 是 LLM↔外部系统的"开放标准通用适配器/插座"：任何模型能插任何工具，不用每次定制对接（对比 CH5 函数调用是专有 1:1）。
# 2. 客户端-服务器架构：LLM → MCP 客户端（宿主 App）→ MCP 服务器（网关）→ 第三方服务（真·API/DB）。服务器按领域暴露能力。
# 3. 三原语：Resource（静态数据，如 PDF/DB 记录）、Tool（可执行函数，如发邮件/查 API）、Prompt（指导 LLM 用资源/工具的模板）。
# 4. 五步流程：① 发现（list_tools 拿清单）→ ② 请求制定（LLM 选工具填参）→ ③ 客户端通信 → ④ 服务器执行 → ⑤ 响应回填上下文。
# 5. MCP vs 函数调用：MCP 标准化、跨模型互操作、动态发现、服务器可复用；函数调用专有、1:1、须显式告知、紧耦合。CH5 是 MCP 的内核。
# 6. 两个坑：① MCP 只是"接口契约"，效果取决于底层 API 设计——别裸包遗留 API，要加确定性能力（过滤/排序）；② 数据格式要对 Agent 友好（PDF→Markdown）。
# 7. 传输层：本地用 STDIO+JSON-RPC，远程用 HTTP/SSE；服务器可本地（快/数据不出机）或远程（组织内共享）。

## ============ 旧代码（Google ADK + Gemini + FastMCP，已注释，仅供对比）============
# 这是原书第10章的"标准写法"：用 ADK 的 MCPToolset 把文件系统/自定义 MCP 服务器当工具挂给 Gemini Agent。
# 优点：ADK 把"发现→连接→调用"全托管了；缺点：绑定 Google ADK + Gemini，无法直接换 DeepSeek。
# from google.adk.agents import LlmAgent
# from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
# from fastmcp import FastMCP
#
# ## --- 书里的 FastMCP 服务器（greet 工具）---
# mcp_server = FastMCP()
# @mcp_server.tool
# def greet(name: str) -> str:
#     """生成个性化的问候语。"""
#     return f"Hello, {name}! Nice to meet you."
#
# ## --- 书里的 ADK 客户端（Gemini 驱动）---
# root_agent = LlmAgent(
#     model='gemini-2.0-flash',
#     name='fastmcp_greeter_agent',
#     instruction='You are a friendly assistant that can greet people by their name. Use the "greet" tool.',
#     tools=[MCPToolset(connection_params=HttpServerParameters(url="http://localhost:8000"), tool_filter=['greet'])],
# )


## ============ 新代码（DeepSeek，当前生效）============
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# FastMCP 是 provider 无关的 MCP 服务器框架（PyPI: pip install fastmcp）。
# 注意：本机若未安装，先执行 `pip install fastmcp mcp` 再跑。
try:
    from fastmcp import FastMCP, Client
except ImportError:
    raise SystemExit("⚠️ 未安装 fastmcp，请先执行：pip install fastmcp mcp")


## --- 1. MCP 服务器（"通往外部的网关"，暴露工具） ---
## 与 CH5 不同：这里的工具不是本地函数直接调，而是作为 MCP 工具"暴露"给任意客户端。
## 用 FastMCP：@mcp.tool 装饰器 + docstring 自动生成工具 schema（含名字/描述/参数）。
mcp_server = FastMCP("ch10-demo")

@mcp_server.tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气（模拟数据，仅用于演示 MCP 工具调用）。

    Args:
        city: 城市名称，例如 "北京"、"上海"、"广州"
    """
    data = {
        "北京": "晴，22°C，西北风3级",
        "上海": "多云，25°C，东南风2级",
        "广州": "雷阵雨，28°C，湿度85%",
    }
    return data.get(city, f"暂无 {city} 的天气数据")

@mcp_server.tool
def get_stock_quote(symbol: str) -> str:
    """获取股票的最新价格（模拟数据，仅用于演示 MCP 工具调用）。

    Args:
        symbol: 股票代码，例如 "600519"（贵州茅台）、"000001"（平安银行）
    """
    data = {
        "600519": "贵州茅台 1488.00 元，今日 +1.2%",
        "000001": "平安银行 11.35 元，今日 -0.3%",
    }
    # 故意：未知代码抛错，用来演示第10章强调的"错误要反馈给模型恢复"（与 CH5 一致）。
    if symbol not in data:
        raise ValueError(f"没有找到股票代码 {symbol} 的行情")
    return data[symbol]


## --- 2. 配置 DeepSeek（作为 Agent 的大脑，替代书里的 Gemini） ---
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


## --- 3. 把 MCP 工具 schema 转成 OpenAI function-calling 格式（bind_tools 要的这种） ---
def _tool_schema(t):
    """尽可能从 MCP/FastMCP 的 Tool 对象里取出 JSON Schema dict。
    兼容不同版本：属性名可能是 inputSchema / input_schema / parameters；
    取值可能是 dict，也可能是 pydantic 模型（需 model_dump）。"""
    for attr in ("inputSchema", "input_schema", "parameters"):
        val = getattr(t, attr, None)
        if val is None:
            continue
        if isinstance(val, dict):
            return val
        if hasattr(val, "model_dump"):   # pydantic v2
            try:
                return val.model_dump()
            except Exception:
                pass
        if hasattr(val, "dict"):         # pydantic v1
            try:
                return val.dict()
            except Exception:
                pass
    return {"type": "object", "properties": {}}

def mcp_tools_to_openai_schema(tools):
    """MCP 的 Tool 对象列表 → OpenAI 的 tools 列表。"""
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "parameters": _tool_schema(t),
            },
        })
    return openai_tools


def _extract_text(result):
    """从 MCP 的 CallToolResult 里抽出文本内容（兼容 .content 列表）。"""
    if hasattr(result, "content"):
        return "\n".join(getattr(c, "text", str(c)) for c in result.content)
    return str(result)


## --- 4. MCP 客户端驱动 + CH5 风格工具循环（核心：如何实现 MCP 调用） ---
async def run_mcp_agent(user_query: str):
    """
    演示第10章"MCP 调用"的完整链路（把 CH5 的本地函数循环换成"远程 MCP 工具"）：
      1. 启动 MCP 客户端，连上服务器（这里用 in-memory 传输，仍走真实 MCP 协议 JSON-RPC）
      2. 发现：client.list_tools() 动态拿工具清单（对应五步流程的①发现）
      3. 把 MCP 工具转成 OpenAI schema，bind 给 DeepSeek（替代 CH5 的本地 @tool 注册）
      4. LLM 决策 + 生成调用（② 请求制定）—— 与 CH5 完全一致
      5. 执行：client.call_tool(name, args) 把调用发到 MCP 服务器（③④ 客户端通信 + 服务器执行）
      6. 回填：把服务器返回结果作为 ToolMessage 回给 LLM（⑤ 响应回填）
      7. LLM 收敛作答
    注：想换成"真·独立服务器"，把 Client(mcp_server) 换成
       Client("http://localhost:8000")（远程 HTTP）或 Client(StdioServerParameters(command=...))（本地 STDIO）即可。
    """
    if not llm:
        print("LLM 未初始化，跳过。")
        return

    # 连服务器（in-memory：客户端与服务器同进程，但通过 MCP 协议 JSON-RPC 通信）
    async with Client(mcp_server) as client:
        mcp_tools = await client.list_tools()                       # ① 发现
        print(f"\n>>> MCP 服务器暴露的工具：{[t.name for t in mcp_tools]}")
        openai_tools = mcp_tools_to_openai_schema(mcp_tools)
        llm_with_tools = llm.bind_tools(openai_tools)

        system = SystemMessage(content=(
            "你是一个助手。用户问天气调用 get_weather，问股票调用 get_stock_quote，"
            "拿到 MCP 工具返回的真实数据后再用中文回答。如果工具报错，基于错误信息向用户说明。"
        ))
        messages = [system, HumanMessage(content=user_query)]

        max_steps = 5  # 防无限调工具的死循环
        for step in range(max_steps):
            print("\n" + "=" * 20 + f" MCP 工具循环：第 {step + 1} 步 " + "=" * 20)

            # --- 2 & 4. LLM 决策 + 生成调用 ---
            ai_msg = llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not ai_msg.tool_calls:
                # --- 7. 模型不再调工具，给出最终回答 ---
                print("\n>>> 模型决定不再调用工具，给出最终回答：")
                print(ai_msg.content)
                break

            # --- 5 & 6. 经 MCP 执行 + 回填 ---
            for call in ai_msg.tool_calls:
                name, args = call["name"], call["args"]
                print(f"\n>>> 阶段：经 MCP 调用工具 {name}({args})")
                try:
                    result = await client.call_tool(name, args)     # ③④ 客户端通信 + 服务器执行
                    text = _extract_text(result)
                    print(f"    MCP 服务器返回：{text}")
                except Exception as e:
                    # 第10章强调的健壮性：工具失败不要崩，把错误反馈给模型让其恢复。
                    text = f"工具执行出错: {type(e).__name__}: {e}"
                    print(f"    ⚠️ MCP 调用失败，已把错误反馈给模型：{text}")
                messages.append(ToolMessage(content=text, tool_call_id=call["id"]))  # ⑤ 回填

    return messages


if __name__ == "__main__":
    asyncio.run(run_mcp_agent("北京今天天气怎么样？另外帮我看下 600519 的股价。"))

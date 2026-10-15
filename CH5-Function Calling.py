## ============ 第5章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 5: Tool Use）============
# 1. 工具使用让 Agent 能跟外部系统交互、访问训练数据之外的动态/实时信息——它是前4章（编排）与真 Agent 的分水岭。
# 2. 工具描述（名字 + 用途 + 参数类型）必须清晰准确：模型完全靠它决定"调不调、调哪个"。
# 3. Function Calling 的 6 步循环：① 工具定义 → ② LLM 决策 → ③ 生成结构化调用 → ④ 框架执行 → ⑤ 结果回填 → ⑥ LLM 再处理（作答/再调/反思）。
# 4. 决定"调不调、调哪个"是模型干的，真正执行是框架干的 → Agent 的"自主"来自循环里模型自己拍板。
# 5. 模型一次可返回多个 tool_calls（并行工具调用）；工具失败时要把错误反馈给模型让其恢复，而非直接崩溃。
# 6. Function Calling（窄，调函数）⊂ Tool Calling（宽，工具可以是 API/DB/另一个 Agent）。

## ============ 旧代码（OpenAI gpt-4o + LangChain AgentExecutor，已注释，仅供对比）============
# 这是原书第5章的"标准写法"：用 create_tool_calling_agent 把"决策→执行→回填"循环交给框架托管。
# 优点：代码短、贴近生产；缺点：循环被框架藏起来，看不清上面那 6 步。
# 注意：它需要额外安装 `langchain`（不只 langchain-core）。
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from langchain.agents import create_tool_calling_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
#
# load_dotenv()
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
#
# @tool
# def get_weather(city: str) -> str:
#     """获取指定城市的当前天气。"""
#     data = {"北京": "晴，22°C", "上海": "多云，25°C"}
#     return data.get(city, f"暂无 {city} 的天气数据")
#
# tools = [get_weather]
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个助手，必要时调用工具。"),
#     ("user", "{input}"),
#     MessagesPlaceholder("agent_scratchpad"),
# ])
# agent = create_tool_calling_agent(llm, tools, prompt)
# executor = AgentExecutor(agent=agent, tools=tools)
# executor.invoke({"input": "北京今天天气怎么样？"})


## ============ 新代码（DeepSeek，当前生效）============
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

## --- 配置 ---
## 使用 DeepSeek（OpenAI 兼容）替代 gpt-4o。
## 检查是否设置了 API 密钥。
if not os.getenv("LEARN_AI_KEY"):
    raise ValueError("在 .env 文件中未找到 LEARN_AI_KEY。请添加它。")

## 初始化聊天 LLM。使用 deepseek-v4-flash。
## 温度设 0：工具调用要确定性，别让模型"随手"发挥。
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


## --- 1. 工具定义（模型靠"名字+用途+参数类型"决定调哪个）---
## 用 @tool 装饰器，函数签名 + docstring 会自动变成模型能理解的 JSON Schema（工具契约）。
## 下面故意放两个工具，用来演示"模型从工具箱里挑选对的那个"（Tool Use 的灵魂）。
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气（模拟数据，仅用于演示 Function Calling 流程）。

    Args:
        city: 城市名称，例如 "北京"、"上海"、"广州"
    """
    # 真实场景这里会调用天气 API；演示用写死的字典即可。
    data = {
        "北京": "晴，22°C，西北风3级",
        "上海": "多云，25°C，东南风2级",
        "广州": "雷阵雨，28°C，湿度85%",
    }
    return data.get(city, f"暂无 {city} 的天气数据")


@tool
def get_stock_quote(symbol: str) -> str:
    """获取股票的最新价格（模拟数据，仅用于演示 Function Calling 流程）。

    Args:
        symbol: 股票代码，例如 "600519"（贵州茅台）、"000001"（平安银行）
    """
    # 演示用写死字典；真实场景这里会调用行情 API。
    data = {
        "600519": "贵州茅台 1488.00 元，今日 +1.2%",
        "000001": "平安银行 11.35 元，今日 -0.3%",
    }
    # 故意：未知代码抛错，用来演示第5章强调的"工具失败要反馈给模型恢复"。
    if symbol not in data:
        raise ValueError(f"没有找到股票代码 {symbol} 的行情")
    return data[symbol]


tools = [get_weather, get_stock_quote]
tools_by_name = {t.name: t for t in tools}

## bind_tools：把工具 schema 挂到模型上，模型才会在需要时报出 tool_calls。
llm_with_tools = llm.bind_tools(tools)


def run_tool_call_loop(user_query: str):
    """
    演示第5章 Function Calling 的 6 步循环（手写版，对应 CH2/CH4 的"保留书结构、只换模型"套路）：
      1. 工具定义   → 已在上面 @tool 完成（两个工具，演示"选择"）
      2. LLM 决策   → 模型看用户请求 + 工具清单，自己决定"调不调、调哪个"
      3. 生成调用   → 要调就输出结构化 tool_calls（工具名 + 参数）
      4. 工具执行   → 编排层拦下 tool_call，真正去跑函数（含错误兜底）
      5. 观察       → 把结果（或错误）作为 ToolMessage 回填给模型
      6. 再处理     → 模型决定"作答 / 再调一个工具 / 反思"（橙色回边收敛）
    """
    system = SystemMessage(content=(
        "你是一个助手。当用户问天气时调用 get_weather，问股票时调用 get_stock_quote，"
        "拿到工具返回的真实数据后再用中文回答。如果工具报错，基于错误信息向用户说明。"
    ))
    messages = [system, HumanMessage(content=user_query)]

    max_steps = 5  # 防止模型陷入无限调工具的死循环
    for step in range(max_steps):
        print("\n" + "=" * 20 + f" 工具循环：第 {step + 1} 步 " + "=" * 20)

        # --- 2 & 3. LLM 决策 + 生成调用 ---
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # --- 6. 模型不再调工具，给出最终回答（橙色回边收敛）---
            print("\n>>> 模型决定不再调用工具，给出最终回答：")
            print(ai_msg.content)
            break

        # --- 4. 工具执行（注意：模型一次可能返回多个 tool_calls = 并行工具调用）---
        for call in ai_msg.tool_calls:
            name = call["name"]
            args = call["args"]
            print(f"\n>>> 阶段：执行工具 {name}({args})")
            try:
                result = tools_by_name[name].invoke(args)
            except Exception as e:
                # 第5章强调的健壮性：工具失败不要崩，把错误作为结果反馈给模型，让它恢复/换策略。
                result = f"工具执行出错: {type(e).__name__}: {e}"
                print(f"    ⚠️ 工具执行失败，已把错误反馈给模型：{result}")
            else:
                print(f"    工具返回：{result}")

            # --- 5. 观察：把结果（或错误）作为 ToolMessage 回填，带 tool_call_id 对齐 ---
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return messages


if __name__ == "__main__":
    # 一句提问同时触发两个工具，演示"模型选择 + 并行工具调用"。
    run_tool_call_loop("北京今天天气怎么样？另外帮我看下 600519 的股价。")

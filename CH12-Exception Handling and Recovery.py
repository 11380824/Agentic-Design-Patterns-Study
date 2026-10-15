## ============ 第12章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 12: Exception Handling and Recovery）============
# 1. 异常处理与恢复 = 让 Agent 在真实世界(工具错/网络抖/脏数据/服务宕)里"不崩、能恢复"，从脆弱系统变可靠组件。
# 2. 三个阶段：① 错误检测（验证工具输出/看 API 错误码/超时）② 错误处理（日志/重试/回退/优雅降级/通知）③ 恢复（状态回滚/诊断/自纠正/升级）。
# 3. 重试(retry)专治"瞬态错误"(如偶发 503)；回退(fallback)是主要路径挂了换替代策略；优雅降级是都挂了也至少给点价值。
# 4. 本模式常与 CH4 反思配合：初始尝试失败→反思分析失败原因→用改进提示词重试。
# 5. 真实用例：数据库宕机的客服机器人、资金不足的交易机器人、坏文件的批处理 Agent(跳过坏的继续)、验证码/404 的爬虫。
# 6. 关键工程点：错误要"可观测"(日志)、瞬态要"可重试"、致命要"可升级给人"——目标是最小化停机、受控失败。

## ============ 旧代码（Google ADK + gemini-2.0-flash，已注释，仅供对比）============
# 原书用 ADK 的 SequentialAgent 串三个子 Agent：primary_handler(调 get_precise_location_info) →
# fallback_handler(看 state["primary_location_failed"] 决定是否用 get_general_area_info) → response_agent(读 state 出结果)。
# 优点是"声明式编排、顺序保证"；缺点是绑定 ADK/Gemini，且错误恢复逻辑藏在 Agent 指令里、不够显式。
# from google.adk.agents import Agent, SequentialAgent
# primary_handler = Agent(name="primary_handler", model="gemini-2.0-flash-exp",
#     instruction="用 get_precise_location_info 获取精确位置。", tools=[get_precise_location_info])
# fallback_handler = Agent(name="fallback_handler", model="gemini-2.0-flash-exp",
#     instruction="若 state['primary_location_failed'] 为 True，从原查询抽城市用 get_general_area_info。", tools=[get_general_area_info])
# response_agent = Agent(name="response_agent", model="gemini-2.0-flash-exp",
#     instruction="读 state['location_result'] 呈现结果；为空则道歉。")
# robust_location_agent = SequentialAgent(name="robust_location_agent",
#     sub_agents=[primary_handler, fallback_handler, response_agent])

## ============ 新代码（DeepSeek，当前生效）============
import os
import time
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


## --- 1. 模拟"外部工具"：可能失败（这就是错误检测的对象）---
PRIMARY_DOWN = True  # 模拟主要位置服务持续宕机（演示回退/降级路径）
_fail_counter = {"n": 0}

def get_precise_location_info(address: str) -> str:
    """主要处理器：精确位置。真实场景会调外部地理 API，这里模拟可能 503 不可用。"""
    _fail_counter["n"] += 1
    if PRIMARY_DOWN:
        raise RuntimeError(f"位置服务 API 返回 503（第 {_fail_counter['n']} 次尝试失败）")
    return f"精确坐标：{address} -> (31.2304, 121.4737)"

def get_general_area_info(city: str) -> str:
    """回退处理器：只给城市级区域信息（降级后的替代策略）。"""
    return f"{city} 所属大致区域：华东，中国"

def notify_human(reason: str):
    """通知/升级机制：严重故障上报给人或上级系统（CH13 人机协同的雏形）。"""
    print(f"    🚨 [通知/升级] {reason}")


## --- 2. 错误检测 + 抽取城市（供回退用）---
def extract_city(query: str) -> str:
    """让 DeepSeek 从用户问题抽城市；llm 不可用时退回简单启发式。"""
    if llm:
        try:
            r = llm.invoke([SystemMessage(content="从用户问题中提取城市名，只返回一个词；没有则回复 NONE。"),
                            HumanMessage(content=query)]).content.strip()
            return "" if r.upper() == "NONE" else r
        except Exception:
            pass
    # 兜底：正则抓"市"前的字
    import re
    m = re.search(r"([\u4e00-\u9fa5]{2,4})市", query)
    return m.group(1) if m else ""


## --- 3. 核心：重试 + 回退 + 优雅降级 的稳健分发（CH12 灵魂）---
def robust_location_dispatch(query: str, max_retries: int = 2) -> str:
    """
    演示第12章完整链路：
      ① 错误检测：每次调工具都用 try 捕获异常（验证输出/捕获 API 错误）
      ② 错误处理-重试：瞬态错误退避后重试（这里 PRIMARY_DOWN 模拟持续故障，重试耗尽）
      ③ 错误处理-回退：主要路径持续失败 → 换 get_general_area_info 替代策略
      ④ 错误处理-优雅降级 + 通知：回退也失败 → 给部分价值(道歉)并升级给人
      ⑤ 恢复：把结果交给 LLM 组织成用户友好的回复（系统回到稳定可操作状态）
    """
    city = extract_city(query)
    print(f"    🔎 抽取城市：{city or '(未识别)'}")

    # ② 主要处理器 + 重试（瞬态错误场景用）
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    ⏳ 尝试主要处理器（第 {attempt} 次）...")
            return "【主要路径】" + get_precise_location_info(query)
        except Exception as e:
            print(f"    ⚠️ 主要处理器出错：{e}")
            if attempt < max_retries:
                time.sleep(0.3)  # 退避，避免对故障服务猛刷

    # ③ 回退处理器
    print("    🔁 主要处理器持续失败，启用回退处理器...")
    if city:
        try:
            return "【回退路径】" + get_general_area_info(city)
        except Exception as e:
            print(f"    ⚠️ 回退处理器也失败：{e}")

    # ④ 优雅降级 + 通知（受控失败，不崩）
    notify_human("精确与区域位置服务均不可用")
    return "【优雅降级】抱歉，暂时无法获取位置信息，请稍后再试或联系人工客服。"


## --- 4. 恢复：用 LLM 把结果组织成自然语言回复 ---
def compose_reply(query: str, raw_result: str) -> str:
    if llm and not raw_result.startswith("【优雅降级】"):
        try:
            return llm.invoke([SystemMessage(content="你是客服助手，用一句中文友好地转述下面的位置结果。"),
                               HumanMessage(content=f"用户问题：{query}\n位置结果：{raw_result}")]).content
        except Exception:
            pass
    return raw_result


if __name__ == "__main__":
    q = "帮我查下上海市中心的确切位置信息"
    print(">>> 用户查询：", q)
    raw = robust_location_dispatch(q)
    print("\n>>> 分发结果：", raw)
    print(">>> 最终回复：", compose_reply(q, raw))
    print("\n提示：把上方 PRIMARY_DOWN 改为 False 可看到'主要路径成功、重试即命中'的瞬态恢复场景。")

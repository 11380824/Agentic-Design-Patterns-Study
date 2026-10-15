## ============ 第13章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 13: Human-in-the-Loop）============
# 1. 人机协同(HITL) = 把人类的判断/创造力/道德推理 与 AI 的计算力结合：AI 干脏活累活，人在关键节点把关。
# 2. 六个关键方面：人类监督 / 干预纠正 / 用于学习的人类反馈(RLHF) / 决策增强(AI 给建议人拍板) / 人机协作 / 升级策略(何时交接给人)。
# 3. 核心机制是"升级(escalation)"：Agent 遇到模糊/高风险/超能力边界的任务，主动暂停交给人，而非擅自硬干。
# 4. 变体"人在环外"(Human-on-the-loop)：人定大策略(如70%科技股)，AI 在策略内高速执行即时操作。
# 5. 代价：可扩展性差(人审不了百万任务) + 依赖高技能专家 + 隐私(敏感信息需匿名化)。
# 6. 真实用例：内容审核(模糊case给人)、自动驾驶(极端天气交还人类)、金融欺诈(高风险警报给人)、法律文件(人终审)。
# 7. 书里 ADK 示例的亮点：用 callback 在调 LLM 前把"客户姓名/等级/历史"注入系统消息做个性化——这本就是 HITL 的"人提供上下文"。

## ============ 旧代码（Google ADK + gemini-2.0-flash，已注释，仅供对比）============
# 原书技术支持 Agent 配了 troubleshoot_issue / create_ticket / escalate_to_human 三个工具，指令里写：
# "复杂问题用 escalate_to_human 转接人类专家"；并用 personalization_callback 在请求 LLM 前把 state 里的客户信息塞进系统消息。
# 优点是"升级工具 + 个性化回调"把 HITL 工程化；缺点是绑定 ADK，且升级动作是'假转接'(只是 return 一句已升级)。
# from google.adk.agents import Agent
# technical_support_agent = Agent(name="technical_support_specialist", model="gemini-2.0-flash-exp",
#     instruction="技术问题先 troubleshoot_issue；持续存在就 create_ticket；复杂的用 escalate_to_human 转人工。",
#     tools=[troubleshoot_issue, create_ticket, escalate_to_human])
# def personalization_callback(cb_ctx, llm_req):
#     ci = cb_ctx.state.get("customer_info")
#     if ci: llm_req.contents.insert(0, system_note(ci))  # 注入客户上下文

## ============ 新代码（DeepSeek，当前生效）============
import os
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


## ============ HITL 配置 ============
# 设为 True：用预置应答自动过人工关卡（脚本/CI 环境可跑，不阻塞）。
# 设为 False：真正调用 input() 等你人工输入（交互式体验更真实）。
AUTO_HUMAN = True
_HUMAN_QUEUE = ["approved", "rejected"]  # AUTO_HUMAN 时依次消耗的模拟人工决策

def request_human_decision(prompt: str) -> str:
    """人工关卡(Human-in-the-Loop 的核心)：把决策权交还人。
    真实系统里这里会推送到人工队列/工单；这里用 input() 或预置应答模拟。"""
    if AUTO_HUMAN:
        decision = _HUMAN_QUEUE.pop(0) if _HUMAN_QUEUE else "approved"
        print(f"    👤 [人工关卡·自动模拟] {prompt} -> 人工决策：{decision}")
        return decision
    try:
        return input(f"    👤 [人工关卡] {prompt} (approved/rejected): ").strip().lower()
    except EOFError:
        # 非交互环境(如管道)兜底，避免卡死
        print(f"    👤 [人工关卡] 非交互环境，默认 approved")
        return "approved"


## --- 1. 工具：Agent 能自己干的 ---
def troubleshoot_issue(issue: str) -> str:
    """基础故障排查（Agent 自主执行）。"""
    return f"排查建议：重启设备、检查网络连接、清除缓存。问题='{issue}'"

def create_ticket(issue: str) -> str:
    """开工单记录（Agent 自主执行）。"""
    return f"已创建工单 TICKET-{hash(issue) % 10000:04d}，等待处理。"

## --- 2. 升级工具：把复杂/高风险问题交给人 ---
def escalate_to_human(issue: str) -> str:
    """升级给人类专家——这是 HITL 的'交接'动作。"""
    decision = request_human_decision(f"是否由人工专家接手处理：{issue}？")
    if decision == "approved":
        return f"✅ 人工专家已接手：{issue}（人类监督生效）"
    return f"⛔ 人工拒绝接手，退回 Agent 自行给出通用方案：{issue}"


## --- 3. 个性化上下文注入（对应书里 personalization_callback）---
def inject_customer_context(query: str, customer_info: dict) -> str:
    """把'人提供的客户上下文'(HITL 的人供给 AI 的信息)拼进提示词。"""
    if not customer_info:
        return query
    note = (f"[客户上下文] 姓名={customer_info.get('name')}，等级={customer_info.get('tier')}，"
            f"近期购买={customer_info.get('recent_purchases')}")
    return f"{note}\n用户问题：{query}"


## --- 4. 主流程：支持 Agent（自主处理 + 复杂则升级）---
def support_agent(query: str, customer_info: dict = None) -> str:
    """演示第13章：Agent 先自主排查，复杂才升级给人；并用客户上下文个性化。"""
    ctx_query = inject_customer_context(query, customer_info)
    print(f"\n>>> 用户问题：{query}")

    # 让 DeepSeek 判断该走哪条路（自主排查 vs 升级）
    router = llm.invoke([SystemMessage(content=(
        "你是技术支持路由。判断用户问题应'自主排查'还是'升级人工'。"
        "仅回复一个词：AUTO(简单可自助) 或 ESCALATE(复杂/高风险需人工)。")),
        HumanMessage(content=ctx_query)]).content.strip().upper()
    print(f"    🧭 路由判断：{router}")

    if router == "ESCALATE":
        return escalate_to_human(query)

    # 自主路径：排查 + 开单
    tip = troubleshoot_issue(query)
    ticket = create_ticket(query)
    return f"{tip}\n{ticket}"


if __name__ == "__main__":
    vip = {"name": "张三", "tier": "VIP", "recent_purchases": ["导航仪X1"]}
    # 场景1：简单问题 → 自主处理
    print(support_agent("我的设备连不上 WiFi", vip))
    # 场景2：复杂/高风险问题 → 升级人工（AUTO_HUMAN 模拟人批准）
    print(support_agent("设备莫名发热并伴随数据泄露风险，请处理", vip))

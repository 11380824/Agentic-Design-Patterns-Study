## ============ 第16章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 16: Resource-Aware Optimization）============
# 1. 资源感知优化 = Agent 运行时动态监控/权衡 计算、时间、钱：在"贵而强"和"便宜而快"之间按任务复杂度/预算/延迟做选择。
# 2. 典型策略：动态模型切换(简单查询→小模型Flash，复杂→大模型Pro)、回退机制(首选模型限流/宕→切默认模型保连续性)、
#    自适应工具选择、上下文修剪/摘要(控 token)、主动资源预测、成本敏感探索、节能部署、并行/分布式、学习型资源分配、优雅降级。
# 3. 常用架构：路由器 Agent 先判复杂度 → 派给合适 LLM/工具；批评 Agent 评估响应质量、反哺路由逻辑。
# 4. 书里三类路由：simple(直接答) / reasoning(多步推理→强模型) / internet_search(需实时→触发搜索)。
# 5. 设计原则：响应质量 vs 运营成本的权衡；把"思考预算"按任务分配，而不是每题都烧最强模型。

## ============ 旧代码（OpenAI + Google CSE，已注释，仅供对比）============
# 原书 classify_prompt 把查询分 simple/reasoning/internet_search，再分别选 gpt-4o-mini / o4-mini / gpt-4o，
# internet_search 还调 Google Custom Search API。优点是"按类选模型"直观；缺点是绑定 OpenAI 三档模型 + Google 搜索密钥。
# def classify_prompt(prompt): ...  # gpt-4o 分类
# def generate_response(prompt, cls, search_results=None):
#     model = {"simple":"gpt-4o-mini","reasoning":"o4-mini","internet_search":"gpt-4o"}[cls]
#     return client.chat.completions.create(model=model, messages=[...]).choices[0].message.content

## ============ 新代码（DeepSeek，当前生效）============
# 设计取舍：我们只配了 deepseek-v4-flash 一个模型，没有"多档模型"可切。所以这里把"资源感知"落在这三件事上：
#   ① 复杂度分类路由(用 v4-flash 自己判 simple/reasoning/search)
#   ② 不同档位用"不同计算策略"模拟(简单=直接答省 token；推理=强制 CoT 多烧思考；搜索=查本地小知识库而非联网)
#   ③ 回退机制(主调用异常→重试→降级到固定模板) + token 成本记账
# 这样在你的真实环境里能直接跑，且把第16章的"按资源分配"思想讲透。
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


## --- 成本记账（对应"跟踪 token 使用量"）---
class CostMeter:
    def __init__(self): self.calls = 0; self.prompt_tokens = 0; self.completion_tokens = 0
    def record(self, usage):
        if not usage: return
        self.calls += 1
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0)
        self.completion_tokens += getattr(usage, "completion_tokens", 0)
    def report(self):
        print(f"  💰 成本记账：调用 {self.calls} 次 | prompt {self.prompt_tokens} tok | completion {self.completion_tokens} tok")
METER = CostMeter()

def _chat(system: str, user: str, **kw):
    if not llm:
        return "(LLM 未初始化)"
    r = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)], **kw)
    METER.record(getattr(r, "usage", None))
    return r.content


## --- 本地小知识库（替代"实时联网搜索"，让 demo 离线可跑）---
KB = {
    "deepseek": "DeepSeek 是幻方量化旗下的 AI 公司，以 MoE 架构和低成本推理著称。",
    "mcp": "MCP(模型上下文协议)是让 LLM 标准化连接外部工具/数据的开放协议。",
    "rag": "RAG 是检索增强生成，先检索外部知识再生成，减少幻觉。",
}

## --- ① 复杂度分类路由（路由器 Agent 的角色）---
def classify(query: str) -> str:
    raw = _chat("你是路由Agent，把用户问题分为三类且仅回一个词：simple / reasoning / search。",
                f"规则：事实回忆→simple；需多步推理/计算→reasoning；问最新/外部知识→search。问题：{query}")
    for tag in ("simple", "reasoning", "search"):
        if tag in raw.lower():
            return tag
    return "simple"

## --- ② 各档位的"计算策略"（模拟资源分配）---
def handle_simple(query: str) -> str:
    """省资源档：直接短答，不烧额外思考 token。"""
    return _chat("用一句话直接回答，不要展开。", query)

def handle_reasoning(query: str) -> str:
    """高资源档：强制思维链(CoT)，多分配思考预算。"""
    return _chat("请一步步思考(Chain-of-Thought)后再给最终答案。", query)

def handle_search(query: str) -> str:
    """搜索档：查本地知识库(替代联网)，把证据拼进提示再生成。"""
    # 极简关键词命中；真实场景换 WebSearch / RAG
    ctx = "\n".join(f"- {k}: {v}" for k, v in KB.items() if k in query.lower())
    if not ctx:
        ctx = "（本地知识库未命中相关条目）"
    return _chat(f"基于资料回答：\n{ctx}", query)


## --- ③ 回退机制：主调用异常→重试→降级 ---
def safe_call(fn, *args, max_retry: int = 2):
    for i in range(max_retry + 1):
        try:
            return fn(*args)
        except Exception as e:
            print(f"    ⚠️ 调用出错({e})，第 {i+1} 次重试...")
    return "【优雅降级】服务暂时不可用，请稍后再试。（已触发回退，未完全失败）"


## --- 主流程：资源感知分发 ---
def resource_aware_answer(query: str) -> str:
    """演示第16章：分类 → 按档位分配计算 → 回退兜底 → 成本记账。"""
    tier = classify(query)
    print(f"  🧭 路由判定档位：{tier}")
    dispatcher = {"simple": handle_simple, "reasoning": handle_reasoning, "search": handle_search}[tier]
    return safe_call(dispatcher, query)


if __name__ == "__main__":
    for q in [
        "法国的首都是哪里？",                       # 期望 → simple（省资源）
        "如果一个水池单开进水管6小时满，单开出水管8小时空，同时开多久满？",  # 期望 → reasoning（多烧思考）
        "请介绍下 RAG 是什么",                       # 期望 → search（查知识库）
    ]:
        print("\n>>> 查询：", q)
        ans = resource_aware_answer(q)
        print(">>> 回答：", ans)
        METER.report()

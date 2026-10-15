## ============ 第20章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 20: Prioritization）============
# 1. 优先级排序 = Agent 按重要性/紧急性/依赖关系/资源/成本收益 给任务/目标排序，把精力放在最关键的事上。
# 2. 四要素：① 标准定义(紧急性/重要性/依赖/资源可用/成本收益/用户偏好) ② 任务评估(规则→LLM 评分) ③ 调度选择(队列/规划)
#    ④ 动态重新优先级(新关键事件出现时实时调整)。
# 3. 排序可发生在三层：总体目标优先级 / 计划内步骤优先级 / 下一个即时动作选择。
# 4. 真实用例：客服(宕机>改密码)、云调度(高峰给关键应用)、自动驾驶(刹车>省油)、交易(高优先快速执行)、
#    项目管理的看板排序、网安(按威胁严重度排警报)、个人助理(按截止日期排提醒)。
# 5. 本质：让 Agent 从"被动执行脚本"变成"主动战略决策者"，类似人类团队的管理者。

## ============ 旧代码（LangChain ReAct 项目管理 Agent，已注释，仅供对比）============
# 原书用 SuperSimpleTaskManager(字典存 Task: id/desc/priority/assigned_to) + create_react_agent，
# 配 PM 系统提示词让 Agent 先 create_new_task 拿 ID，再按语义分配 P0/P1/P2 和 Worker，最后 list_all_tasks。
# 优点是"LLM 自主解释模糊请求并选工具"；缺点是绑定 LangChain AgentExecutor + gpt-4o-mini，且每次都走 ReAct 循环。
# llm = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
# pm_agent = create_react_agent(llm, pm_tools, pm_prompt_template)

## ============ 新代码（DeepSeek，当前生效）============
# 这里把第20章的"优先级排序"落为可跑 demo：用 DeepSeek 给任务打 P0/P1/P2，维护一个按优先级排序的队列，
# 并演示"动态重新优先级"——突发高优事件(系统宕机)插入后自动顶到最前。
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


## --- 任务模型 + 优先级权重 ---
_PRIORITY_WEIGHT = {"P0": 3, "P1": 2, "P2": 1}
class Task:
    _seq = 0
    def __init__(self, desc: str, priority: str):
        Task._seq += 1
        self.id = f"TASK-{Task._seq:03d}"
        self.desc = desc
        self.priority = priority
    def __repr__(self):
        return f"{self.id} [{self.priority}] {self.desc}"

class PriorityQueue:
    def __init__(self): self.tasks = []
    def add(self, task: Task):
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: _PRIORITY_WEIGHT.get(t.priority, 0), reverse=True)
    def show(self, title: str = "当前队列"):
        print(f"  📋 {title}（高→低）：")
        for t in self.tasks:
            print(f"     {t}")


## --- 用 DeepSeek 给新任务定优先级（任务评估 + 调度选择）---
def classify_priority(desc: str) -> str:
    if not llm:
        return "P1"
    r = llm.invoke([SystemMessage(content=(
        "你是工单路由。按紧急性/重要性把任务定为 P0(最高)/P1(中)/P2(低) 之一，仅回一个词。")),
        HumanMessage(content=desc)]).content.strip().upper()
    return r if r in ("P0", "P1", "P2") else "P1"

def add_task(pq: PriorityQueue, desc: str, force_priority: str = None):
    pri = force_priority or classify_priority(desc)
    t = Task(desc, pri)
    pq.add(t)
    print(f"  ➕ 新增 {t}（优先级由{'指定' if force_priority else 'LLM 评估'}）")
    return t


if __name__ == "__main__":
    pq = PriorityQueue()
    # 初始几个任务（让 LLM 评估优先级）
    for desc in [
        "审查营销网站内容",
        "为 VIP 客户上线新登录系统（很紧急）",
        "整理上周周报",
    ]:
        add_task(pq, desc)
    pq.show("初始队列")

    # 动态重新优先级：突发"生产环境宕机"——这是最高优，强制 P0 插队
    print("\n⚡ 突发：生产环境数据库宕机！触发动态重新优先级...")
    add_task(pq, "生产环境数据库宕机，立即抢修", force_priority="P0")
    pq.show("重新排序后（宕机应顶到最前）")

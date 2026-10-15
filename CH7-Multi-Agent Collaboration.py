## ============ 第7章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 7: Multi-Agent Collaboration）============
# 1. 多智能体协作 = 把复杂目标拆给多个各有专长的 Agent 分工完成；威力来自"劳动分工 + 智能体间通信"。
# 2. 6 种协作形态：顺序交接 / 并行处理 / 辩论共识 / 层次结构(主管+员工) / 专家团队 / 批评者-审查者(=CH4反思的多Agent版)。
# 3. 关系/通信光谱 6 级：单Agent → 网络(P2P) → 监督者 → 监督者作为工具 → 层次化 → 自定义混合。
# 4. 多Agent 是"容器"不是新技术：CH2路由→监督者委派专家；CH3并行→多Agent并行；CH4反思→批评者审查者；
#    CH5工具→ADK 的 "Agent as Tool"(把一个Agent包成工具调，工具调用的升级版)；CH6规划→顺序交接(每步一个Agent)。
# 5. 代价：更多 token 成本、Agent 间协调开销、调试更难；任务能拆成"需要不同专业技能/工具"的子任务才上多Agent。
# 6. 何时该用：单 Agent 搞不定的多领域任务（复杂研究、软件开发、创意内容、客服升级）。不该用：单 Agent 能搞定的别硬上。
# 7. 和你架构的对应：你的"CC生成 + codebuddy/DSH双评审" = 批评者-审查者(专家团队)形态；若加个协调器分派任务即"层次结构(监督者)"。

## ============ 旧代码（CrewAI + OpenAI gpt-4 / Google ADK，已注释，仅供对比）============
# 原书第7章标准写法：
#  CrewAI: researcher + writer 两个 Agent 顺序跑（专家团队 / 顺序交接形态）
#  Google ADK: 演示父子层次、LoopAgent(循环)、SequentialAgent、ParallelAgent、AgentTool(Agent 当工具)
# 注意：需额外装 crewai / google-adk，底层是 gpt-4，国内需换 DeepSeek。
# from crewai import Agent, Task, Crew, Process
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4", temperature=0)
# researcher = Agent(role="高级研究报告员", goal="研究主题并整理要点", backstory="你擅长信息检索与归纳。", llm=llm, verbose=True)
# writer = Agent(role="技术内容作家", goal="基于研究要点写文章", backstory="你文笔清晰、面向技术团队。", llm=llm, verbose=True)
# research_task = Task(description="研究主题：{topic}", agent=researcher)
# write_task = Task(description="基于研究要点写文章", agent=writer, context=[research_task])
# crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task], process=Process.sequential)
# crew.kickoff(inputs={"topic": "多智能体协作的价值"})


## ============ 新代码（DeepSeek，当前生效）============
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

## --- 配置 ---
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


def call_agent(role_prompt: str, task: str) -> str:
    """
    多 Agent 的本质：同一个 LLM，靠不同 SystemMessage 扮演不同"角色"。
    这就是第7章的核心——角色分离（不是换模型）。你的"CC生成 + 双评审"同模型不同前端，正是这种思路。
    """
    label = role_prompt.split("。")[0].replace("你是", "").strip()
    print(f"\n>>> [Agent: {label}] 处理任务")
    sys = SystemMessage(content=role_prompt)
    res = llm.invoke([sys, HumanMessage(content=task)])
    print(f"    产出(节选)：{res.content[:100]}...")
    return res.content


def researcher_agent(topic: str) -> str:
    """专家角色①：研究员——收集/整理信息（专家团队形态）。"""
    return call_agent(
        "你是高级研究报告员。针对给定主题，整理 3-5 条关键事实/要点，简洁列出，不要写长文。",
        f"主题：{topic}",
    )


def writer_agent(topic: str, research: str) -> str:
    """专家角色②：作家——基于研究要点写最终内容（专家团队形态）。"""
    return call_agent(
        "你是技术内容作家。基于提供的研究要点，写一篇 150 字左右、面向技术团队的中文短文。",
        f"主题：{topic}\n研究要点：\n{research}",
    )


def critic_agent(draft: str, topic: str) -> str:
    """评审者 Agent：批判评估初稿（批评者-审查者形态 = CH4 反思的多 Agent 版，对应你的"CC生成 + 双评审"）。"""
    return call_agent(
        "你是严格的技术评审员。指出下面初稿的 2-3 个具体改进点（事实/逻辑/清晰度的不足），每条一句话。",
        f"主题：{topic}\n初稿：\n{draft}",
    )


def orchestrator(topic: str) -> str:
    """
    主管 Agent（Orchestrator）：把任务委派给专家、综合结果。
    这是第7章"层次结构(监督者)"形态的简化版——同一 LLM、不同角色提示词。
    演示两种形态叠用：专家团队(研究员→作家) + 批评者-审查者(评审→作家修订)。
    """
    print("\n" + "=" * 20 + " 主管 Agent：委派子任务 " + "=" * 20)
    # 形态① 专家团队 / 顺序交接
    research = researcher_agent(topic)
    draft = writer_agent(topic, research)
    # 形态⑥ 批评者-审查者（= CH4 反思的多 Agent 版）
    feedback = critic_agent(draft, topic)
    final = writer_agent(f"{topic}（请根据评审意见修订：{feedback}）", research)
    return final


if __name__ == "__main__":
    # 跑起来你会看到：主管把任务分给"研究员→作家"顺序协作，再经"评审者"批判、"作家"据反馈修订。
    # 这正是第7章的"专家团队 + 批评者-审查者"叠用，也完全对应你正在搭的"生成 + 双评审"多 Agent 结构。
    # 想试其他形态：把 critic_agent 换成两个研究员"辩论共识"，或把 researcher/writer 用 CH3 的并行方式同时跑。
    result = orchestrator("多智能体协作在 AI Agent 开发中的价值")
    print("\n" + "=" * 20 + " 最终交付 " + "=" * 20)
    print(result)

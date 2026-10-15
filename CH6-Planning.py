## ============ 第6章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 6: Planning）============
# 1. 规划 = 把复杂高层目标拆成可管理的小步骤、制定从初始状态到目标状态的行动序列。
# 2. 计划是"动态生成"的，不是你写死的——你只给目标和约束，方法由模型自己设计。
# 3. 规划 vs 提示链(CH1)：提示链步骤顺序是你写死的（被动编排）；规划是模型自主生成步骤（主动决策）。Agent 的"自主性"差在这一下。
# 4. 规划站在工具调用(CH5)之上：子任务靠调工具落地；子任务常复用前几章（并行CH3/路由CH2/反思CH4）。
# 5. 适应性是灵魂：初始计划只是起点，环境变化（场地不可用/数据缺失）时模型记录新约束、重新规划而非失败。
# 6. 权衡：解法"已知且可重复"用写死工作流更可靠（限制自主权=减少不确定性）；解法"需被发现"才上规划。明确提示"先列计划"会鼓励此行为。
# 7. 标杆系统：Google Gemini Deep Research（先给你审阅修改再执行）、OpenAI Deep Research API（o3 + web_search，带内联引用报告）。

## ============ 旧代码（CrewAI + OpenAI gpt-4-turbo，已注释，仅供对比）============
# 原书第6章"标准写法"：用 CrewAI 单 Agent 先建要点计划、再写摘要，Process.sequential 顺序跑。
# 优点：贴近生产、代码短；缺点：计划生成与执行被框架藏起，看不清下面手写的"计划→执行→评审→重规划"循环。
# 注意：它需要额外安装 `crewai`，且底层是 gpt-4-turbo，国内需换 DeepSeek。
# from crewai import Agent, Task, Crew, Process
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
# planner_writer = Agent(
#     role="内容策划与写作者",
#     goal="针对给定主题先制定结构化要点计划，再基于计划撰写摘要",
#     backstory="你擅长把模糊需求拆解成清晰步骤，再高质量执行。",
#     llm=llm, verbose=True,
# )
# task = Task(
#     description="主题：{topic}。请先列出 3-5 条要点计划，再据此写一段中文摘要。",
#     agent=planner_writer,
# )
# crew = Crew(agents=[planner_writer], tasks=[task], process=Process.sequential)
# crew.kickoff(inputs={"topic": "远程团队协作的最佳实践"})


## ============ 新代码（DeepSeek，当前生效）============
import os
import re
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


def parse_numbered_steps(text: str) -> list[str]:
    """从模型输出里抽出编号步骤（1. / 2、/ 3) 都认）。"""
    steps = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[\.\、\)]\s*(.+)$", line)
        if m:
            steps.append(m.group(1).strip())
    return steps


def generate_plan(goal: str) -> list[str]:
    """阶段① 规划：把高层目标拆成可执行的子步骤（动态生成，非写死）。"""
    print("\n" + "=" * 20 + " 阶段①：生成计划 " + "=" * 20)
    sys = SystemMessage(content=(
        "你是一个任务规划器。给定一个目标，请拆成 3-5 个有序子步骤。"
        "只输出编号列表，每行一个步骤，例如：\n1. 第一步\n2. 第二步"
    ))
    msg = llm.invoke([sys, HumanMessage(content=f"目标：{goal}")])
    steps = parse_numbered_steps(msg.content)
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    return steps


def execute_step(step: str, goal: str, done: list[str]) -> str:
    """阶段② 执行：让模型"完成"某个子步骤（演示用 LLM 模拟执行；
    真实场景这里会调工具落地，见 CH5 的 run_tool_call_loop）。"""
    print(f"\n>>> 执行子步骤：{step}")
    sys = SystemMessage(content=(
        "你是执行者。基于已有进展，完成下面这一子步骤，用 1-2 句话给出该步骤的具体产出/结论。"
    ))
    ctx = f"总目标：{goal}\n已完成步骤及产出：\n" + ("\n".join(f"- {d}" for d in done) or "（无）")
    res = llm.invoke([sys, HumanMessage(content=f"{ctx}\n\n当前子步骤：{step}")])
    print(f"    产出：{res.content}")
    return res.content


def review(goal: str, done: list[str]) -> tuple[bool, str]:
    """阶段③ 评审：判断目标是否达成；不达成则返回缺失项（用于适应性重规划）。"""
    print("\n" + "=" * 20 + " 阶段③：评审是否达成目标 " + "=" * 20)
    sys = SystemMessage(content=(
        "你是评审员。判断目标是否已基于已完成步骤的产出达成。"
        "回复严格按此格式：\n状态: 完成\n或\n状态: 未完成\n缺失: <还差什么，一句话>"
    ))
    ctx = f"总目标：{goal}\n已完成步骤及产出：\n" + ("\n".join(f"- {d}" for d in done) or "（无）")
    res = llm.invoke([sys, HumanMessage(content=ctx)])
    text = res.content
    print(text)
    done_flag = ("状态" in text) and ("完成" in text) and ("未完成" not in text)
    missing = ""
    if not done_flag:
        m = re.search(r"缺失[:：]\s*(.+)", text)
        missing = m.group(1).strip() if m else "（未说明具体缺失）"
    return done_flag, missing


def run_planning_loop(goal: str, max_iter: int = 3):
    """
    演示第6章 Planning 的核心循环（手写版，对齐 CH4/CH5 风格）：
      ① 规划  → 模型把目标拆成子步骤（动态生成）
      ② 执行  → 逐个子步骤让模型产出结果（真实场景此处调工具，见 CH5）
      ③ 评审  → 模型判断是否达成；不达成 → 带"缺失项"回到 ① 重规划（适应性）
    循环直到"完成"或达到 max_iter。
    """
    done: list[str] = []
    missing = ""
    for it in range(max_iter):
        print("\n" + "#" * 60)
        print(f"# 规划循环：第 {it + 1} 轮")
        print("#" * 60)

        # ① 规划（若上一轮有缺失项，作为新约束带进本轮，演示"适应性"）
        full_goal = goal
        if missing:
            full_goal += f"\n（上一轮评审指出还缺：{missing}。请在新计划里补上这一步。）"
        plan = generate_plan(full_goal)

        # ② 执行（逐个子步骤产出结果，累积到 done）
        for step in plan:
            done.append(execute_step(step, goal, done))

        # ③ 评审：达成就停，否则进入下一轮重规划
        ok, missing = review(goal, done)
        if ok:
            print("\n✅ 目标达成，规划循环结束。")
            break
        else:
            print(f"\n🔄 目标未达成，进入第 {it + 2} 轮重规划（适应性演示）。")

    return done


if __name__ == "__main__":
    # 一个"解法需要被发现"的开放目标，正好适合规划模式。
    # 跑起来你会看到：① 模型自己列步骤 → ② 逐个执行 → ③ 评审不达标 → 带缺失项重规划。
    run_planning_loop("为团队组织一次线下技术分享会，输出可落地的筹备清单")

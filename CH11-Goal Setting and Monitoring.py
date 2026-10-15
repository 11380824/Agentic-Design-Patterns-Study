## ============ 第11章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 11: Goal Setting and Monitoring）============
# 1. 目标设定与监控 = 给 Agent 两样东西：明确的目标 + 判断"成没成功"的监控回路，把反应式系统升级成"主动冲目标"的系统。
# 2. 它是 CH4(反思) + CH6(规划) 之上的元模式：规划管"怎么拆步骤"，反思管"生成→改"，CH11 再加"显式目标清单 + 达标判定(True/False)"驱动迭代/重规划/升级。
# 3. 目标要 SMART（具体/可衡量/可实现/相关/有时限）；必须定义清楚的指标与成功标准，监控才有效。
# 4. 监控 = 观察 Agent 行动 + 环境状态 + 工具输出，对比目标得出偏差，反馈回路让 Agent 适应/重规划/升级。
# 5. 书里编码示例：接收用例+质量目标清单 → 生成 → 自我评审 → LLM 判 True/False → False 带反馈修订、True/到上限停 → 落盘 .py。
# 6. 书里自承的坑：同一 LLM 既写又判有偏、可能幻觉式误判达标、不会魔法产出完美代码(仍要跑测试)、基础监控有无限循环风险(需 max_iter 兜底)。
# 7. 书的建议：拆成多 Agent 团队(程序员/审查员/文档/测试/提示词优化)让独立裁判当评委，客观性↑——与你"多 Agent 单 LLM"现状对应，质量天花板仍在底层模型。

## ============ 旧代码（OpenAI gpt-4o + LangChain，已注释，仅供对比）============
# 这是原书第11章的"标准写法"：单 LLM(gpt-4o) 既写代码又当裁判。核心骨架：
#   generate_prompt(use_case, goals, previous_code, feedback)  -> 拼生成/修订提示词
#   get_code_feedback(code, goals)                            -> LLM 当审查员写批评
#   goals_met(feedback_text, goals)                           -> LLM 只答 True/False 判定达标
#   run_code_agent(use_case, goals_input, max_iterations=5)   -> 生成→评审→判定→修订 循环，落盘 .py
# 优点：逻辑直观、贴合生产雏形；缺点：写+判同一模型有偏（书里自己承认），且绑定 gpt-4o。
# （完整约 190 行，此处仅保留骨架注释；完整版见 adp.xindoo Chapter 11）

## ============ 新代码（DeepSeek，当前生效）============
import os
import re
import random
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
        temperature=0.2,
        api_key=os.getenv("LEARN_AI_KEY"),
        base_url="https://api.deepseek.com",
    )
    print(f"语言模型已初始化: {getattr(llm, 'model', 'deepseek-v4-flash')}")
except Exception as e:
    print(f"初始化语言模型时出错: {e}")
    llm = None


## --- 1. 生产者：生成 / 修订（目标驱动的"写代码"角色） ---
GENERATOR_SYSTEM = SystemMessage(content="你是一个严谨的 Python 编码 Agent，必须按给定的目标清单产出代码。")

def generate_prompt(use_case, goals, previous_code="", feedback=""):
    """拼装生成或修订代码的提示词：用例 + 目标清单 + (上版代码) + (反馈)。"""
    prompt = f"""你是一个 AI 编码 Agent。根据以下用例编写 Python 代码：

用例：{use_case}

你的目标（必须满足）：
{chr(10).join(f"- {g.strip()}" for g in goals)}
"""
    if previous_code:
        prompt += f"\n之前生成的代码：\n{previous_code}"
    if feedback:
        prompt += f"\n对之前版本的反馈（请据此修订）：\n{feedback}"
    prompt += "\n请仅返回修订后的 Python 代码，不要包含代码块外的解释。"
    return prompt


## --- 2. 监控器第一步：评审（指出差距，不判定） ---
REVIEWER_SYSTEM = SystemMessage(content="""你是一名资深 Python 代码审查员。基于给定的目标清单，逐条评估代码：
- 指出逻辑错误、运行时风险、未处理的边缘情况；
- 评估清晰度、简单性、正确性；
- 如果全部目标都满足，明确说"目标已全部满足"；
否则列出仍需改进的点。""")

def get_code_feedback(code, goals):
    """监控器的"观察+评估"：对照目标产出批评（这一步只指出差距，不下达标结论）。"""
    user = f"目标清单：\n{chr(10).join(f'- {g.strip()}' for g in goals)}\n\n代码：\n{code}"
    return llm.invoke([REVIEWER_SYSTEM, HumanMessage(content=user)]).content


## --- 3. 监控器第二步：达标判定（True/False，驱动停/续） ---
JUDGE_SYSTEM = SystemMessage(content="你是一个严格、客观的 AI 评审员。只依据反馈与目标的匹配度回答。")

def goals_met(feedback_text, goals):
    """监控器的"判定"：让 LLM 只答 True/False，作为停条件（对应 CH11 的'监控判定'）。
    容错：只要含 true 且不含 false 才视为达标，避免模型幻觉式误判导致过早停。"""
    prompt = f"""目标清单：
{chr(10).join(f'- {g.strip()}' for g in goals)}

关于代码的反馈：
\"\"\"
{feedback_text}
\"\"\"

根据上述反馈，所有目标是否已达成？仅用一个词回答：True 或 False。"""
    resp = llm.invoke([JUDGE_SYSTEM, HumanMessage(content=prompt)]).content.strip().lower()
    return "true" in resp and "false" not in resp


## --- 4. 工具函数：清理代码块 / 加注释头 / 落盘 ---
def clean_code_block(code):
    """去掉 LLM 可能包的反引号代码围栏。"""
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def add_comment_header(code, use_case):
    """加用例注释头，方便人读。"""
    return f"# 此 Python 程序实现以下用例：\n# {use_case.strip()}\n\n" + code

def save_code_to_file(code, use_case):
    """用用例 slug + 随机后缀生成文件名落盘（省一次 LLM 取名调用）。"""
    short = re.sub(r"[^a-z0-9_]", "", use_case.lower().replace(" ", "_"))[:10]
    fname = f"{short}_{random.randint(1000, 9999)}.py"
    Path(fname).write_text(code, encoding="utf-8")
    print(f"✅ 代码已保存：{fname}")
    return fname


## --- 5. 主循环：目标设定 + 监控驱动（CH11 核心） ---
def run_goal_agent(use_case, goals_input, max_iterations=5):
    """
    演示第11章"目标设定与监控"完整链路：
      1. 设定目标：把用例 + 质量目标清单交给 Agent（SMART 化的成功标准）
      2. 生成：生产者按目标写第一版代码
      3. 监控-评估：审查员系统对照目标找差距（观察 Agent 产出 vs 目标）
      4. 监控-判定：评审员只答 True/False → 达标就停，未达标带反馈回去修订（反馈回路）
      5. 重复 2~4，直到 True 或触达 max_iterations（无限循环兜底）
      6. 落盘：达标后加注释头、存成 .py
    注：这里用"单 LLM + 角色分离"(生成/审查/判定三种 SystemMessage) 模拟书里建议的
        "独立裁判"；真要更客观可拆成 CH7 多 Agent（生成者/审查员用不同模型），但质量天花板仍在底层模型。
    """
    if not llm:
        print("LLM 未初始化，跳过。")
        return

    goals = [g.strip() for g in goals_input.split(",") if g.strip()]
    print(f"\n🎯 用例：{use_case}")
    print("🎯 目标：")
    for g in goals:
        print(f"  - {g}")

    previous_code, feedback = "", ""
    for i in range(max_iterations):
        print(f"\n=== 迭代 {i + 1}/{max_iterations} ===")

        # 2. 生成 / 修订
        prompt = generate_prompt(use_case, goals, previous_code, feedback)
        raw = llm.invoke([GENERATOR_SYSTEM, HumanMessage(content=prompt)]).content
        code = clean_code_block(raw)
        print("\n🧾 生成代码：\n" + "-" * 50 + f"\n{code}\n" + "-" * 50)

        # 3. 监控-评估
        feedback = get_code_feedback(code, goals)
        print("\n📥 评审反馈：\n" + "-" * 50 + f"\n{feedback}\n" + "-" * 50)

        # 4. 监控-判定（停条件）
        if goals_met(feedback, goals):
            print("✅ 监控判定：目标已达成，停止迭代。")
            break
        print("🛠️ 目标未完全达成，准备下一轮修订...")
        previous_code = code
    else:
        # 循环自然跑完（未被 break）才执行：无限循环兜底
        print("⚠️ 达到最大迭代次数仍未完全达标（已触发无限循环兜底）。")

    final = add_comment_header(code, use_case)
    return save_code_to_file(final, use_case)


if __name__ == "__main__":
    # 示例：书里同款 BinaryGap 题，目标清单即"成功标准"
    use_case = "编写代码查找给定正整数的 BinaryGap（二进制表示中，首尾 1 之间最长连续 0 的位数）"
    goals = "代码简单易懂,功能正确,处理边缘情况(如输入0或全1应返回0),仅接受正整数,打印结果并附带示例"
    run_goal_agent(use_case, goals, max_iterations=5)

## ============ 第4章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 4: Reflection）============
# 1. 反思 = 生成 → 自我/他者评审 → 基于批评迭代完善，直到达标或到上限——目标是"提质"。
# 2. 两种形态：① 自我反思（单 Agent，自己审自己）；② 生产者-审查者（多 Agent，生成者 + 独立批判者）。
# 3. 不等于 ReAct：反思是"产出后回头改"，ReAct 是"边想边调工具边行动"的交错循环（思考-行动交替）。
# 4. 停止条件要明确（如 'CODE_IS_PERFECT' 或 max_iterations），否则会无限循环。
# 5. 质量天花板 = 底层模型本身；CH4 的循环只是给模型"多次机会"，不突破模型能力上限（与你多 Agent 单 LLM 的结论一致）。

## ============ 旧代码（OpenAI gpt-4o，已注释，仅供对比）============
# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage, HumanMessage
#
# ## --- 配置 ---
# ## 从 .env 文件加载环境变量（用于 OPENAI_API_KEY）
# load_dotenv()
#
# ## 检查是否设置了 API 密钥
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("在 .env 文件中未找到 OPENAI_API_KEY。请添加它。")
#
# ## 初始化聊天 LLM。我们使用 gpt-4o 以获得更好的推理。
# ## 使用较低的温度以获得更确定性的输出。
# llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

## ============ 新代码（DeepSeek，当前生效）============
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

## --- 配置 ---
## 使用 DeepSeek（OpenAI 兼容）替代 gpt-4o
## 检查是否设置了 API 密钥
if not os.getenv("LEARN_AI_KEY"):
    raise ValueError("在 .env 文件中未找到 LEARN_AI_KEY。请添加它。")

## 初始化聊天 LLM。使用 deepseek-v4-flash 以获得更好的推理。
## 使用较低的温度以获得更确定性的输出。
try:
    llm = ChatOpenAI(
        model="deepseek-v4-flash",
        temperature=0.1,
        api_key=os.getenv("LEARN_AI_KEY"),
        base_url="https://api.deepseek.com",
    )
    print(f"语言模型已初始化: {getattr(llm, 'model', 'deepseek-v4-flash')}")
except Exception as e:
    print(f"初始化语言模型时出错: {e}")
    llm = None


def run_reflection_loop():
    """
    演示多步 AI 反思循环以逐步改进 Python 函数。
    """
    # --- 核心任务 ---
    task_prompt = """
    你的任务是创建一个名为 `calculate_factorial` 的 Python 函数。
    此函数应执行以下操作：
    1. 接受单个整数 `n` 作为输入。
    2. 计算其阶乘 (n!)。
    3. 包含清楚解释函数功能的文档字符串。
    4. 处理边缘情况：0 的阶乘是 1。
    5. 处理无效输入：如果输入是负数，则引发 ValueError。
    """

    # --- 反思循环 ---
    max_iterations = 3
    current_code = ""

    # 我们将构建对话历史以在每一步中提供上下文。
    message_history = [HumanMessage(content=task_prompt)]

    for i in range(max_iterations):
        print("\n" + "=" * 25 + f" 反思循环：迭代 {i + 1} " + "=" * 25)

        # --- 1. 生成/完善阶段 ---
        # 在第一次迭代中，它生成。在后续迭代中，它完善。
        if i == 0:
            print("\n>>> 阶段 1：生成初始代码...")
            # 第一条消息只是任务提示词。
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> 阶段 1：基于先前批评完善代码...")
            # 消息历史现在包含任务、
            # 最后一个代码和最后一个批评。
            # 我们指示模型应用批评。
            message_history.append(HumanMessage(content="请使用提供的批评完善代码。"))
            response = llm.invoke(message_history)
            current_code = response.content

        print("\n--- 生成的代码 (v" + str(i + 1) + ") ---\n" + current_code)
        message_history.append(response)  # 将生成的代码添加到历史记录

        # --- 2. 反思阶段 ---
        print("\n>>> 阶段 2：对生成的代码进行反思...")
        # 为反思智能体创建特定提示词。
        # 这要求模型充当高级代码审查员。
        reflector_prompt = [
            SystemMessage(content="""
                你是一名高级软件工程师和 Python 专家。
                你的角色是执行细致的代码审查。
                根据原始任务要求批判性地评估提供的 Python 代码。
                查找错误、风格问题、缺失的边缘情况和改进领域。
                如果代码完美并满足所有要求，
                用单一短语 'CODE_IS_PERFECT' 响应。
                否则，提供批评的项目符号列表。
            """),
            HumanMessage(content=f"原始任务：\n{task_prompt}\n\n要审查的代码：\n{current_code}")
        ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        # --- 3. 停止条件 ---
        if "CODE_IS_PERFECT" in critique:
            print("\n--- 批评 ---\n未发现进一步批评。代码令人满意。")
            break

        print("\n--- 批评 ---\n" + critique)

        # 将批评添加到历史记录以用于下一个完善循环。
        message_history.append(HumanMessage(content=f"对先前代码的批评：\n{critique}"))

    print("\n" + "=" * 30 + " 最终结果 " + "=" * 30)
    print("\n反思过程后的最终精炼代码：\n")
    print(current_code)


if __name__ == "__main__":
    run_reflection_loop()
## ============ 第1章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 1: Prompt Chaining）============
# 1. 提示词链把复杂任务拆成多个顺序步骤，每步一个专用 prompt，前一步输出喂给下一步输入——"分而治之 + 顺序串联"。
# 2. 适合"可预测、步骤固定"的流水线（如 提取→转换→格式化），链路是写死的线性流程，不动态跳转。
# 3. 单 Agent 可用、多 Agent 也可用：本质是对"步骤"的编排，与实体数量无关（区别于 CH7 多 Agent）。
# 4. 每步可配不同 prompt/模型/温度，逐步降低单步难度，提升可控性与可调试性。
# 5. 局限：线性写死，无法按中间结果动态分派——那是 CH2 路由 / CH6 规划解决的（对比见各章要点）。

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

## 为了更好的安全性，从 .env 文件加载环境变量
## from dotenv import load_dotenv
## load_dotenv()
## 确保你的 OPENAI_API_KEY 在 .env 文件中设置

## 初始化语言模型（推荐使用 ChatOpenAI）
#llm = ChatOpenAI(temperature=0)
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("LEARN_AI_KEY"),
    base_url="https://api.deepseek.com",
)

## --- 提示词 1：提取信息 ---
prompt_extract = ChatPromptTemplate.from_template(
    "从以下文本中提取技术规格：\n\n{text_input}"
)

## --- 提示词 2：转换为 JSON ---
prompt_transform = ChatPromptTemplate.from_template(
    "将以下规格转换为 JSON 对象，使用 'cpu'、'memory' 和 'storage' 作为键：\n\n{specifications}"
)

## --- 利用 LCEL 构建处理链 ---
## StrOutputParser() 将 LLM 的消息输出转换为简单字符串。
extraction_chain = prompt_extract | llm | StrOutputParser()

## 完整的链将提取链的输出传递到转换提示词的 'specifications' 变量中。
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

## --- 运行链 ---
input_text = "新款笔记本电脑型号配备 3.5 GHz 八核处理器、16GB 内存和 1TB NVMe 固态硬盘。"

## 使用输入文本字典执行链。
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- 最终 JSON 输出 ---")
print(final_result)
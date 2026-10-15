"""
第8章 核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 8: Memory Management）
1. 记忆让 Agent 超越"一次性问答"；分短期(本次会话内)与长期(跨会话持久)两套。
2. 短期记忆 = LLM 上下文窗口(最近消息/工具结果/反思)；窗口有限，满了要"总结旧片段"腾地方。
3. 长期记忆 = 模型外部存储(向量库/知识图谱/DB)，需要时检索进来再注入短期上下文。
4. 长上下文模型只是把窗口拉大，仍是临时的；真要持久必须靠长期记忆。
5. LangChain 三分法：语义记忆(事实/偏好)→用户 profile JSON；情景记忆(经历)→少样本示例；
   程序记忆(规则)→系统提示，可经 CH4 反思后自我改进。
6. 长期记忆的"检索"通常就是一个工具调用(接 CH5)——本文件用 save/recall 两个工具演示。
7. Google ADK 三件套：Session(线程) + State(线程内临时字典) + MemoryService(长期可搜索知识库)。
8. 对你 ESP32 导览设备的映射：短期=本次游览的 GPS/信标/展品上下文；长期=用户画像(喜好/历史轨迹)，
   BLE 触发时检索注入——本文件长期记忆实现可直接套到那个项目。
"""

## ============ 旧代码（Google ADK / LangGraph，已注释，仅供对比）============
# 原书第8章标准写法：
#  Google ADK: 用 Session 承载一次对话线程，State 存线程内临时数据(字典，前缀 user:/app:/temp: 控制归属)，
#              MemoryService 提供跨会话可搜索的长期知识库（通常通过一个"记忆工具"存取）。
#  LangGraph: 用 InMemorySaver / SqliteSaver 做短期 checkpoint，用 Store(namespaced key + 向量相似度) 做长期。
# from google.adk.sessions import Session
# from langgraph.checkpoint.sqlite import SqliteSaver   # 注意：这里 SQLite 是"短期 checkpoint"，不是长期记忆
# 注意：需额外装 google-adk / langgraph，底层 Gemini，国内需换 DeepSeek。


## ============ 新代码（DeepSeek，当前生效）============
import os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # 加载 myProject/.env 中的 LEARN_AI_KEY

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

# ---------- 配置 ----------
if not os.getenv("LEARN_AI_KEY"):
    raise ValueError("在 .env 文件中未找到 LEARN_AI_KEY。")
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("LEARN_AI_KEY"),
    base_url="https://api.deepseek.com",
)
print(f"语言模型已初始化: deepseek-v4-flash")

# 长期记忆存哪？——用 JSON 文件（不是 SQLite）。理由见 LongTermMemory 类文档，文末也有说明。
LTM_PATH = Path(__file__).resolve().parent / "ch8_longterm_memory.json"


# ============ 短期记忆：内存里的消息列表 + 满了自动总结 ============
class ShortTermMemory:
    """
    短期记忆 = 本次会话的上下文窗口。
    - 不持久（会话结束即丢，符合"短期"定义）。
    - 超过 max_keep 条就把最旧的压缩成一条 SystemMessage 摘要，腾出窗口（对应要点②）。
    """
    def __init__(self, max_keep: int = 6):
        self.msgs: list = []
        self.max_keep = max_keep

    def add(self, m) -> None:
        self.msgs.append(m)

    def to_langchain(self) -> list:
        return self.msgs

    def maybe_compact(self) -> None:
        if len(self.msgs) <= self.max_keep:
            return
        keep = max(1, self.max_keep // 2)            # 至少保留最近的几条
        old = self.msgs[: len(self.msgs) - keep]
        recent = self.msgs[len(self.msgs) - keep:]

        # 修复被总结"拆散"的工具调用配对：
        # 1) old 里 AI 消息带 tool_calls 的 id 集合（它们的工具结果若进了 recent 就成孤儿）
        old_tool_call_ids = set()
        for m in old:
            tcs = getattr(m, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    old_tool_call_ids.add(tc["id"])

        # 2) 丢弃孤儿 ToolMessage（其 AI 来源已被总结掉）
        recent = [
            m for m in recent
            if not (getattr(m, "type", None) == "tool" and m.tool_call_id in old_tool_call_ids)
        ]

        # 3) 反向：保留的 AI 消息若所有 tool_call 都没有对应 ToolMessage 了，也丢弃它
        kept_tool_ids = {
            m.tool_call_id for m in recent
            if getattr(m, "type", None) == "tool"
        }
        recent = [
            m for m in recent
            if not (
                getattr(m, "type", None) == "ai" and getattr(m, "tool_calls", None)
                and not any(tc["id"] in kept_tool_ids for tc in m.tool_calls)
            )
        ]

        summary = _summarize(old)
        self.msgs = [SystemMessage(content="[历史对话摘要] " + summary)] + recent
        print(f"    [短期记忆压缩] 旧对话已总结为 1 条摘要，当前窗口 {len(self.msgs)} 条")


# ============ 长期记忆：JSON 文件持久化 + 关键词检索（教学占位） ============
class LongTermMemory:
    """
    长期记忆 = 跨会话持久的用户知识（语义记忆）。
    实现选择：JSON 文件（不是 SQLite）。理由：
      - 零新增依赖（json 是标准库），与 CH1-CH7 依赖一致；
      - 教材第8章明确说"语义记忆 = 用户 profile JSON"，文件最贴合；
      - 教学上肉眼可读，比数据库表直观。
    检索：此处用"关键词重叠"打分做占位。真实生产应做 embedding + 向量库语义检索（如 Chroma/FAISS），
          但那会引入新依赖、超出本课范围，这里如实标注。
    """
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {"facts": []}
        return {"facts": []}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_fact(self, fact: str) -> None:
        self.data.setdefault("facts", []).append(fact)
        self._save()
        print(f"    [长期记忆] 已持久化事实：{fact}")

    def recall(self, query: str, top_k: int = 3) -> list:
        """关键词重叠打分（占位）。真实应做向量语义相似度。"""
        q = set(query.lower().split())
        scored = []
        for f in self.data.get("facts", []):
            score = len(q & set(f.lower().split()))
            if score > 0:
                scored.append((score, f))
        scored.sort(reverse=True)
        return [f for _, f in scored[:top_k]]


# ============ 长期记忆通过"工具"存取（接 CH5：检索本身就是一个工具调用）============
ltm = LongTermMemory(LTM_PATH)

@tool
def save_user_fact(fact: str) -> str:
    """当用户透露了关于自己的长期有用信息（偏好/经历/身份/目标）时调用，把它存进长期记忆，供以后个性化使用。"""
    ltm.save_fact(fact)
    return "已存入长期记忆"

@tool
def recall_user_facts(query: str) -> str:
    """回答需要了解用户历史或偏好时调用，从长期记忆检索相关事实。"""
    facts = ltm.recall(query)
    return " | ".join(facts) if facts else "（无相关长期记忆）"

tools = [save_user_fact, recall_user_facts]
llm_with_tools = llm.bind_tools(tools)


def _summarize(old_msgs: list) -> str:
    text = "\n".join(f"{type(m).__name__}: {getattr(m, 'content', '')}" for m in old_msgs)
    r = llm.invoke([
        SystemMessage(content="把以下对话浓缩成 3 句以内的中文摘要，重点保留用户偏好与关键事实。"),
        HumanMessage(content=text),
    ])
    return r.content


def run_turn(memory: ShortTermMemory, user_input: str, max_steps: int = 6) -> str:
    """
    一轮对话：短期记忆(上下文) + 长期记忆(经工具检索注入) + CH5 工具调用循环。
    自主程度：模型自己决定"要不要存/要不要回忆长期记忆、调哪个工具"，对应要点⑥。
    """
    memory.add(HumanMessage(content=user_input))

    # 长期记忆检索：把相关事实注入系统提示（对应要点③"检索进来再注入短期上下文"）
    remembered = ltm.recall(user_input)
    longterm_hint = "已知用户长期记忆：" + ("；".join(remembered) if remembered else "（暂无）")

    for _ in range(max_steps):
        memory.maybe_compact()
        sys = SystemMessage(content="你是一个博物馆导览助手，语气亲切。" + longterm_hint)
        resp = llm_with_tools.invoke([sys] + memory.to_langchain())
        memory.add(resp)
        if not resp.tool_calls:
            return resp.content
        for tc in resp.tool_calls:
            fn = next((t for t in tools if t.name == tc["name"]), None)
            out = fn.invoke(tc["args"]) if fn else f"未知工具: {tc['name']}"
            memory.add(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return "(达到最大步数，仍未收敛)"


if __name__ == "__main__":
    # ---- 会话1：用户透露偏好，Agent 存入长期记忆 ----
    print("\n" + "=" * 20 + " 会话1（短期+长期记忆写入）" + "=" * 20)
    mem1 = ShortTermMemory()
    run_turn(mem1, "我特别喜欢印象派画作，周末经常去博物馆。")
    run_turn(mem1, "下次来能给我推荐点什么吗？")   # 模型可能调用 recall/直接基于已存事实回答

    # ---- 会话2：全新短期记忆(旧上下文已丢)，但长期记忆文件还在 ----
    print("\n" + "=" * 20 + " 会话2（短期记忆重置，长期记忆跨会话保留）" + "=" * 20)
    mem2 = ShortTermMemory()                       # 全新短期记忆，模拟"新开一次对话"
    answer = run_turn(mem2, "今天有什么适合我的展览？")  # 模型应主动 recall 长期记忆，给出个性化推荐
    print("\n>>> 最终回答：", answer)

    # ---- 短期记忆压缩演示：多轮后窗口超出阈值，自动总结 ----
    print("\n" + "=" * 20 + " 短期记忆压缩演示" + "=" * 20)
    mem3 = ShortTermMemory(max_keep=4)
    for i in range(8):
        run_turn(mem3, f"随便聊第 {i+1} 句，关于艺术的话题。")

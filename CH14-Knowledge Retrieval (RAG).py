## ============ 第14章核心要点（来自 adp.xindoo 《Agentic Design Patterns》Chapter 14: Knowledge Retrieval / RAG）============
# 1. RAG = 检索增强生成：LLM 生成前先去"外部知识库"查相关片段，拼进提示词再生成，把闭卷考生变开卷考生。
# 2. 三步：检索(retrieve，语义搜最相关块) → 增强(augment，把块拼进 prompt) → 生成(generate，基于证据作答)。
# 3. 核心价值：突破训练数据静态上限（实时/私有/专业数据）、减少幻觉、答案可溯源(引用来源)。
# 4. 关键概念：嵌入(Embedding，文本→高维向量) / 语义相似度(向量距离) / 分块(chunking) / 向量数据库 / BM25+向量混合检索。
# 5. 进阶：Agentic RAG = 在检索上加一层"推理 Agent"主动验真/协调冲突/多步子查询/识别知识缺口调外部工具；GraphRAG 用知识图谱跨文档综合。
# 6. 挑战：信息分散在多块难综合、分块/检索质量决定上限、知识库需定期更新(延迟+成本)。

## ============ 旧代码（Google ADK / LangChain + Weaviate，已注释，仅供对比）============
# 原书三例：① ADK 用 google_search 工具直接 RAG；② VertexAiRagMemoryService 接 GCP RAG Corpus；
# ③ LangChain+LangGraph 完整链路：TextLoader 加载 → CharacterTextSplitter 分块 → OpenAIEmbeddings 嵌入 →
#    Weaviate 向量库 → StateGraph(retrieve → generate) 编译运行。优点是"生产级全链路"；缺点是绑定 GCP/Weaviate、
#    且检索依赖 OpenAIEmbeddings(需联网+密钥)，离线跑不了。
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Weaviate
# vectorstore = Weaviate.from_documents(client=client, documents=chunks, embedding=OpenAIEmbeddings(), by_text=False)
# retriever = vectorstore.as_retriever()

## ============ 新代码（DeepSeek + 纯 Python 检索，当前生效）============
# 设计取舍：向量数据库(Weaviate/FAISS)和真实 Embedding 需要额外安装/联网。为让你"开箱即跑"，
# 检索部分用【纯 Python 的 TF-IDF + 余弦相似度】实现(零依赖、离线可跑)，只把"生成"交给 DeepSeek。
# 这仍是标准 RAG（检索增强生成）；真要语义级检索，把 retrieve() 换成 embedding 模型即可（注释里给出替换点）。
import os
import math
from pathlib import Path
from collections import defaultdict
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


## --- 1. 知识库（演示用：公司 WiFi/设备政策小语料）---
KNOWLEDGE_BASE = [
    "员工 WiFi 连接步骤：打开设置-无线局域网，选择 CorpNet，输入工牌号+密码，首次需短信验证码。",
    "设备发热处理规范：停止充电、置于通风处、温度超 45°C 立即停机并报修，禁止继续高负载运行。",
    "数据泄露上报流程：发现疑似泄露须在 1 小时内邮件通知安全部，并冻结相关账号权限。",
    "远程办公 VPN 配置：下载官方客户端，用域账号登录，选择'总部'节点，禁止用第三方代理绕行。",
    "工单升级规则：VIP 客户问题 30 分钟内必须升级到二线；普通客户 4 小时内响应即可。",
    "访客网络策略：访客连 GuestNet，限速 10Mbps，禁止访问内网与打印服务，每日 0 点重置。",
]


## --- 2. 纯 Python 检索器：TF-IDF + 余弦相似度（零依赖，对应"检索"步）---
_STOP = set("的 了 和 在 是 我 你 他 它 与 及 并 或 对 为 从 到 一个 可以 不 请 用 需 要".split())

def _tokenize(text: str):
    # 中英文都很朴素的分词：英文按单词、中文按单字，去停用词
    toks = []
    buf = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                toks.append("".join(buf)); buf = []
            if "\u4e00" <= ch <= "\u9fa5":
                toks.append(ch)
    if buf:
        toks.append("".join(buf))
    return [t for t in toks if t and t not in _STOP]

def _build_tfidf(docs):
    df = defaultdict(int)
    for d in docs:
        for t in set(_tokenize(d)):
            df[t] += 1
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}
    vecs = []
    for d in docs:
        tf = defaultdict(int)
        for t in _tokenize(d):
            tf[t] += 1
        length = sum(tf.values()) or 1
        vec = {t: (tf[t] / length) * idf[t] for t in tf}
        vecs.append(vec)
    return vecs, idf

def _cosine(a, b):
    keys = set(a) | set(b)
    num = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db + 1e-9)

_DOC_VECS, _IDF = _build_tfidf(KNOWLEDGE_BASE)

def retrieve(query: str, k: int = 2):
    """检索：把 query 向量化，返回与知识库余弦相似度最高的 k 个块。
    【替换点】要做语义级检索，把这里换成 embedding 模型：
        vec = embed(query); sims = [cosine(vec, embed(doc)) for doc in KNOWLEDGE_BASE]"""
    qvec = {}
    tf = defaultdict(int)
    for t in _tokenize(query):
        tf[t] += 1
    for t, c in tf.items():
        if t in _IDF:
            qvec[t] = c * _IDF[t]
    scored = sorted(range(len(KNOWLEDGE_BASE)),
                    key=lambda i: _cosine(qvec, _DOC_VECS[i]), reverse=True)
    return [(i, _cosine(qvec, _DOC_VECS[i])) for i in scored[:k]]


## --- 3. 生成：把检索到的块"增强"进 prompt，交给 DeepSeek（对应"增强+生成"步）---
def rag_answer(query: str, k: int = 2):
    """演示第14章 RAG 完整链路：检索 → 增强 → 生成。"""
    hits = retrieve(query, k)
    context = "\n".join(f"[{i+1}] (相似度 {s:.2f}) {KNOWLEDGE_BASE[i]}" for i, s in hits)
    print(f"\n🔍 检索到 {len(hits)} 个相关片段：\n{context}")

    prompt = f"""你是企业知识助手。只能基于下面的【参考资料】回答，不得编造。
如果资料不足以回答，明确说"资料中未提及"。回答后附上引用的片段编号。

【参考资料】
{context}

用户问题：{query}
回答："""
    if not llm:
        return "（LLM 未初始化）" + context
    ans = llm.invoke([SystemMessage(content="严谨、基于证据作答，中文。"),
                     HumanMessage(content=prompt)]).content
    return ans


if __name__ == "__main__":
    # 这句问法跟知识库字面不完全一致("连不上网" vs "WiFi 连接步骤")，TF-IDF 仍能靠共享词检索到——体现 RAG 价值
    q = "我的设备连不上公司的无线网，该怎么办？"
    print(">>> 问题：", q)
    print("\n>>> 回答：", rag_answer(q))

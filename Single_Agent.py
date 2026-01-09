import os

import re
from functools import lru_cache
from typing import Dict, Any
from pathlib import Path
from typing import List, Dict, Any
import json
from langchain_openai import ChatOpenAI  # pip install langchain-openai
from config import LLM_ROUTES,LLM_key
import os

# 示例一：使用 LangChain 的 ChatTongyi
try:
    from langchain_community.chat_models import ChatTongyi
    USE_TONGYI = True
except ImportError:
    USE_TONGYI = False
# # 构建阿里云百炼大模型客户端
# llm = ChatTongyi(
#     model="qwen-turbo",
#     api_key="sk-f61048a35d2e4c08a64696b07e555b5e",
# )

# # 示例二：构建 deepseek 的大语言模型
# from langchain_deepseek import ChatDeepSeek
# llm = ChatDeepSeek(
#     model="deepseek-chat",
#     api_key="sk-f653b7558f204594b0367a4c8de9575f",       #需在AI模型的的官网上申请，例如sk-XXXXX
# )



# # 示例三：构建 腾讯混元 的大语言模型
# import os
# from langchain_openai import ChatOpenAI
# # ====== 配置腾讯混元 API ======
# # 腾讯混元 API 文档：https://cloud.tencent.com/document/product/1729
# os.environ["OPENAI_API_KEY"] = "<API Key>"  # 替换为腾讯云控制台获取的 API Key
# os.environ["OPENAI_API_BASE"] = "https://api.hunyuan.cloud.tencent.com/v1"  # 腾讯混元 OpenAI 兼容地址
#
# llm = ChatOpenAI(
#     model="hunyuan-lite",  # 模型名称，可换成 hunyuan-pro/hunyuan-standard 等
#     temperature=0.7,
#     max_tokens=512
# )


# # 示例四 使用本地安装的qwen2.5模型执行推理
# from langchain_ollama import ChatOllama
# # 初始化模型
# llm = ChatOllama(
#     # model="qwen3:4b",  # 必须包含完整模型名称和tag
#     model="qwen2.5:latest",  # 必须包含完整模型名称和tag
#     # model="deepseek-r1:14b",  # 必须包含完整模型名称和tag
#     # model="qwen2.5:32b",  # 必须包含完整模型名称和tag
#     base_url="http://localhost:11434",  # Ollama默认地址
#     num_gpu=999,  # 使用所有GPU层
#     num_thread=8,   # CPU线程数
#     temperature=0.7,
#     num_ctx=32768,  # 上下文长度
# )

def get_llm(llm_name:str="deepSeek") -> ChatOpenAI:
    """返回一个通用对话模型，用于分类和学科评估"""
    if "tongyi" in llm_name:
        # 示例一：使用 LangChain 的 ChatTongyi
        from langchain_community.chat_models import ChatTongyi
        # 构建阿里云百炼大模型客户端
        llm = ChatTongyi(
            model="qwen-turbo",
            api_key=LLM_key.get("tongyi"),
        )
    elif "deepSeek" in llm_name:
        # 示例二：构建 deepseek 的大语言模型
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=LLM_key.get("deepSeek"),  # 需在AI模型的的官网上申请，例如sk-XXXXX
        )
    elif "hunyuan" in llm_name:
        # 示例三：构建 腾讯混元 的大语言模型
        from langchain_openai import ChatOpenAI
        # 腾讯混元 API 文档：https://cloud.tencent.com/document/product/1729
        os.environ["OPENAI_API_KEY"] = LLM_key.get("hunyuan")  # 替换为腾讯云控制台获取的 API Key
        os.environ["OPENAI_API_BASE"] = "https://api.hunyuan.cloud.tencent.com/v1"  # 腾讯混元 OpenAI 兼容地址
        llm = ChatOpenAI(
            model="hunyuan-lite",  # 模型名称，可换成 hunyuan-pro/hunyuan-standard 等
            temperature=0.7,
            max_tokens=512
        )
    elif "local" in llm_name:
        # 示例四 使用本地安装的qwen2.5模型执行推理
        from langchain_ollama import ChatOllama
        # 初始化模型
        llm = ChatOllama(
            # model="qwen3:4b",  # 必须包含完整模型名称和tag
            model="qwen2.5:latest",  # 必须包含完整模型名称和tag
            # model="deepseek-r1:14b",  # 必须包含完整模型名称和tag
            base_url="http://localhost:11434",  # Ollama默认地址
            num_gpu=999,  # 使用所有GPU层
            num_thread=8,   # CPU线程数
            temperature=0.7,
            num_ctx=32768,  # 上下文长度
        )
    else:
        # 示例一：使用 LangChain 的 ChatTongyi
        from langchain_community.chat_models import ChatTongyi
        # 构建阿里云百炼大模型客户端
        llm = ChatTongyi(
            model="qwen-turbo",
            api_key=LLM_key.get("tongyi"),
        )
    return llm

def call_llm(messages,llm:str="deepSeek") -> str:
    """
    一个统一的大模型调用封装。
    参数 messages 形如：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    返回：大模型回复的纯文本（str）
    """

    if USE_TONGYI:
        resp = get_llm().invoke(messages)
        # typeRes = resp.content
        # typeMess = messages[1]
        # print({"call_llm原文": f"{typeMess}"})
        # print({"call_llm": f"问题结果：{typeRes}"})
        return resp.content
    else:
        # 如果没装 ChatTongyi，这里先抛异常，避免你忘记配置
        raise RuntimeError("请先安装 langchain_community 并配置 ChatTongyi，"
                           "或在 call_llm 中改为你自己的大模型调用。")

def call_llm_P(messages) -> str:
    """
    一个统一的大模型调用封装。
    参数 messages 形如：
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
    返回：大模型回复的纯文本（str）
    """

    if USE_TONGYI:
        resp = get_llm().invoke(messages)
        # typeRes = resp.content
        # typeMess = messages[1]
        # print({"call_llm原文": f"{typeMess}"})
        # print({"call_llm": f"问题结果：{typeRes}"})
        return resp
    else:
        # 如果没装 ChatTongyi，这里先抛异常，避免你忘记配置
        raise RuntimeError("请先安装 langchain_community 并配置 ChatTongyi，"
                           "或在 call_llm 中改为你自己的大模型调用。")


# ========== 3. 2～7 号智能体：【各学科】采选 Agent ==========
def evaluate_book_for_category(book_text: str, category: str) -> Dict[str, Any]:
    """
    按学科分类调用大模型，让它给出是否采纳 + 推荐分数（-1~1）+ 理由。
    为了方便解析，强制要求它输出“三行格式”。

    返回 dict:
    {
        "accept": True/False,
        "score": float between -1 and 1,
        "reason": str
    }
    """
    # 不同学科可以有稍微不同的采选偏好，你可以后续精细化这个提示词
    policy_text = load_subject_policy(category)  # 读取对应学科分类的规则提示词
    system_prompt = f"""你是山东师范大学图书馆资源建设部的采选专家，
    负责“{category}”图书的资源建设工作。
    请结合学校学科布局、人才培养和科研需求，判断这本书是否值得购入馆藏。

    【本学科的采选偏好/说明】：
    {policy_text}

    【待评估图书信息】：
    {book_text}

    请你从本学科的视角出发，综合判断本书是否适合本校采购。
    先给出你的“直觉判断”，再交由系统规则进行二次权重调整。
    # 【输出要求（非常重要）】
    你必须严格按照下面的“三行格式”输出，不要添加任何多余内容：

    第一行：接受 或 拒绝
    第二行：LLM_推荐分数：x
    第三行：一句话理由

    其中：
    - “接受”表示建议购入；
    - “拒绝”表示不建议购入；
    - LLM_推荐分数 x 是 -1 到 1 之间的小数（可以带小数点），
      越接近 1 表示越推荐，越接近 -1 表示越不推荐。"""


    user_prompt = f"下面是候选图书的信息：\n{book_text}\n\n请你做出判断并按规定格式输出："

    resp = call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    lines = [line.strip() for line in resp.strip().splitlines() if line.strip()]

    # 默认值，防止解析失败
    accept = False
    score = 0.0
    reason = ""

    # 解析第一行：接受 / 拒绝
    if len(lines) >= 1:
        if "接受" in lines[0]:
            accept = True
        elif "拒绝" in lines[0]:
            accept = False

    # 解析第二行：推荐分数
    if len(lines) >= 2:
        match = re.search(r"-?\d+(\.\d+)?", lines[1])
        if match:
            try:
                score = float(match.group())
                # 限制在 [-1, 1] 范围内
                score = max(-1.0, min(1.0, score))
            except ValueError:
                score = 0.0

    # 解析第三行：理由
    if len(lines) >= 3:
        reason = lines[2]
    elif len(lines) == 2:
        reason = lines[1]

    return {
        "accept": accept,
        "score": score,
        "reason": reason
    }
# 1. 学科名 -> 提示词文件名映射
SUBJECT_POLICY_FILES: Dict[str, str] = {
    "人文科学类": "humanities.txt",
    "社会科学类": "social_science.txt",
    "理工类": "stem.txt",
    "语言教育类": "language_edu.txt",
    "艺术类": "arts.txt",
    "其他类": "other.txt",
}
# 2. 提示词文件根目录（相对于项目根目录，可按需要调整）
SUBJECT_POLICY_DIR = Path("subject_policies")
@lru_cache(maxsize=None)
def load_subject_policy(subject_name: str) -> str:
    filename = SUBJECT_POLICY_FILES.get(subject_name, "other.txt")
    path = SUBJECT_POLICY_DIR / filename

    base = ""
    if path.exists():
        base = path.read_text(encoding="utf-8")
    else:
        base = f"你是山东师范大学“{subject_name}”方向的图书采访专家，请从研究型高校角度审查图书学术与教学价值。"

    # ✅ 追加全局人工反馈（若存在）
    feedback_path = SUBJECT_POLICY_DIR / "_human_feedback_global.txt"
    if feedback_path.exists():
        fb = feedback_path.read_text(encoding="utf-8")
        base = base + "\n\n" + fb

    return base
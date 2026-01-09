import os

import re
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from typing import List, Dict, Any
import json

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI  # pip install langchain-openai
from config import LLM_ROUTES,LLM_key
import os

# 示例一：使用 LangChain 的 ChatTongyi
try:
    from langchain_community.chat_models import ChatTongyi
    USE_TONGYI = True
except ImportError:
    USE_TONGYI = False


def get_llm(
    llm_name: str = "deepSeek",
    *,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 512,
    timeout: Optional[float] = 60,
    streaming: bool = False,
    # 给 OpenAI-compatible / Ollama 等预留扩展参数
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    统一入口：传入模型名称即可返回可用的 chat model（用于 LangGraph 节点里 llm.invoke）。
    """

    provider, model = _split_provider(llm_name)
    low = (llm_name or "").lower()

    # ------------- 1) provider 自动推断（兼容原来的 deepSeek/tongyi/hunyuan/local 写法） -------------
    if provider is None:
        # 兼容旧写法：包含关键字即走对应 provider（就是这么写的）:contentReference[oaicite:3]{index=3}
        if "tongyi" in low or "qwen" in low:
            provider = "tongyi"
            model = "qwen-turbo" if model == llm_name else model
        elif "deepseek" in low:
            provider = "deepseek"
            model = "deepseek-chat" if model == llm_name else model
        elif "hunyuan" in low:
            provider = "hunyuan"
            model = "hunyuan-lite" if model == llm_name else model
        elif low in {"local", "ollama"} or "ollama" in low:
            provider = "ollama"
            model = "qwen2.5:latest" if model == llm_name else model
        # 直接传 OpenAI 模型名（gpt-4o / gpt-3.5-turbo）
        elif low.startswith("gpt-") or low.startswith("o1-") or low.startswith("chatgpt-"):
            provider = "openai"
        # 直接传 llama / 本地 tag（llama3.1 / qwen2.5:latest）
        elif ("llama" in low) or (":" in model and "/" not in model):
            provider = "ollama"
        else:
            # 兜底：默认用 OpenAI（也可以改成 tongyi）
            provider = "openai"

    provider = provider.lower()

    # ------------- 2) OpenAI 官方（GPT-3.5 / GPT-4o / GPT-4.1 ...） -------------
    if provider in {"openai", "oai"}:
        from langchain_openai import ChatOpenAI
        api_key = _get_secret("openai", "OPENAI_API_KEY")
        # 可选：自建网关时也可以走这个
        final_base_url = base_url or _get_secret("openai_base_url", "OPENAI_BASE_URL")

        if not api_key and not final_base_url:
            raise EnvironmentError("未配置 OPENAI_API_KEY（或 OPENAI_BASE_URL/openai_base_url）。")

        params = dict(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            streaming=streaming,
            **kwargs,
        )
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if final_base_url:
            params["base_url"] = final_base_url

        return ChatOpenAI(**params)

    # ------------- 3) DeepSeek -------------
    if provider in {"deepseek"}:
        from langchain_deepseek import ChatDeepSeek

        api_key = _get_secret("deepSeek", "deepseek", "DEEPSEEK_API_KEY", "DEEPSEEK_KEY")
        if not api_key:
            raise EnvironmentError("未配置 deepseek 的 key（deepSeek/DEEPSEEK_API_KEY）。")

        params = dict(model=model or "deepseek-chat", api_key=api_key, temperature=temperature, **kwargs)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return ChatDeepSeek(**params)

    # ------------- 4) 通义千问 Tongyi/Qwen -------------
    if provider in {"tongyi", "qwen"}:
        from langchain_community.chat_models import ChatTongyi

        api_key = _get_secret("tongyi", "TONGYI_API_KEY", "DASHSCOPE_API_KEY")
        if not api_key:
            raise EnvironmentError("未配置 tongyi 的 key（tongyi/TONGYI_API_KEY/DASHSCOPE_API_KEY）。")

        # ChatTongyi 不同版本对 timeout/streaming 参数兼容不一，尽量少塞
        params = dict(model=model or "qwen-turbo", api_key=api_key, temperature=temperature, **kwargs)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return ChatTongyi(**params)

    # ------------- 5) 腾讯混元（OpenAI-compatible，避免修改 os.environ 全局变量） -------------
    if provider in {"hunyuan"}:
        from langchain_openai import ChatOpenAI

        api_key = _get_secret("hunyuan", "HUNYUAN_API_KEY")
        if not api_key:
            raise EnvironmentError("未配置 hunyuan 的 key（hunyuan/HUNYUAN_API_KEY）。")

        final_base_url = base_url or "https://api.hunyuan.cloud.tencent.com/v1"

        params = dict(
            model=model or "hunyuan-lite",
            api_key=api_key,
            base_url=final_base_url,
            temperature=temperature,
            timeout=timeout,
            streaming=streaming,
            **kwargs,
        )
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return ChatOpenAI(**params)

    # ------------- 6) 本地/内网 Ollama（Llama/Qwen/DeepSeek-R1 等本地模型都能跑） -------------
    if provider in {"ollama", "local"}:
        from langchain_ollama import ChatOllama

        final_base_url = base_url or _get_secret("ollama_base_url", "OLLAMA_BASE_URL", "OLLAMA_HOST") or "http://localhost:11434"

        params = dict(
            model=model or "qwen2.5:latest",
            base_url=final_base_url,
            temperature=temperature,
            num_gpu=999,  # 使用所有GPU层
            num_thread=8,  # CPU线程数
            num_ctx=32768,  # 上下文长度
            **kwargs,
        )
        # ChatOllama 常用 num_predict 对应 max_tokens
        if max_tokens is not None and "num_predict" not in params:
            params["num_predict"] = max_tokens
        return ChatOllama(**params)

    # ------------- 7) OpenAI-Compatible 通用入口（一个口子打通很多平台） -------------
    # 例如 openrouter/groq/together/fireworks/vllm/oneapi/siliconflow 等
    if provider in {"openai_compat", "compat", "openrouter", "groq", "together", "fireworks", "vllm", "oneapi", "siliconflow"}:
        from langchain_openai import ChatOpenAI

        # 约定：LLM_key 里可以放 {provider}_base_url / {provider}，也可用环境变量
        final_base_url = base_url or _get_secret(f"{provider}_base_url", f"{provider.upper()}_BASE_URL", "OPENAI_COMPAT_BASE_URL")
        api_key = _get_secret(provider, f"{provider.upper()}_API_KEY", "OPENAI_COMPAT_API_KEY", "API_KEY")

        if not final_base_url:
            raise EnvironmentError(f"未配置 {provider} 的 base_url（{provider}_base_url / {provider.upper()}_BASE_URL）。")

        params = dict(
            model=model,
            api_key=api_key or "EMPTY",
            base_url=final_base_url,
            temperature=temperature,
            timeout=timeout,
            streaming=streaming,
            **kwargs,
        )
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return ChatOpenAI(**params)

    raise ValueError(f"不支持的 llm provider: {provider}（llm_name={llm_name}）")

def _get_secret(*candidates: str) -> Optional[str]:
    """优先从 LLM_key 取，其次从环境变量取。"""
    for k in candidates:
        if not k:
            continue
        v = None
        # config dict
        try:
            v = LLM_key.get(k)
        except Exception:
            v = None
        if v:
            return v
        # env
        v = os.getenv(k)
        if v:
            return v
    return None

def _split_provider(name: str) -> Tuple[Optional[str], str]:
    """
    支持两种传参：
      1) 仅模型名： "gpt-4o" / "llama3.1" / "qwen2.5:latest"
      2) provider:model： "openai:gpt-4o" / "ollama:llama3.1" / "openrouter:xxx"
    """
    s = (name or "").strip()
    if ":" in s:
        p, m = s.split(":", 1)
        # 注意：ollama 的 model 经常也含 ':'（tag），例如 qwen2.5:latest
        # 所以我们只把“第一个冒号”当 provider 分隔符，后面的属于 model
        if p.lower() in {
            "openai", "oai", "deepseek", "tongyi", "qwen", "hunyuan",
            "ollama", "local",
            "openai_compat", "compat",
            "openrouter", "groq", "together", "fireworks", "vllm", "oneapi", "siliconflow"
        }:
            return p.strip().lower(), m.strip()
    return None, s

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
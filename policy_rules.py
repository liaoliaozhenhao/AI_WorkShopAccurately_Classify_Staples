#基于 学校采访政策 的 LLM 采选评判

import os
import re
import math
import pickle
import json  # 新增
from typing import Union, Optional
import numpy as np
import pandas as pd
import Single_Agent

SLLM=Single_Agent
def load_policy_text(path: str) -> str:
    """读取采选政策文本，用于给 LLM 做 system prompt。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_book_policy_payload(row: pd.Series) -> dict:
    """
    把单本图书的信息整理成一个精简 dict，用于 LLM 判断。
    可以根据你的表结构再增减字段。
    """
    return {
        "题名": row.get("题名", ""),
        "责任者": row.get("责任者", ""),
        "出版社": row.get("出版社", ""),
        "出版年": row.get("出版年", ""),
        "价格": row.get("价格", ""),
        "分类号": row.get("分类号", ""),
        "文献类型": row.get("文献类型", ""),
        "作品语种": row.get("作品语种", ""),
        "内容简介": row.get("内容简介", ""),
        "学科大类": row.get("subject_category", ""),
        "分类前缀": row.get("call_prefix", ""),
        "馆藏需求分": float(row.get("need_overall_score", 0.0)),
        "模型分数(score_model)": float(row.get("score_model", 0.0)),
        "规则分(rule_pos_score)": int(row.get("rule_pos_score", 0)),
        "规则惩罚(rule_softneg_score)": int(row.get("rule_softneg_score", 0)),
        "硬规则触发": int(row.get("rule_hard_reject", 0)),
    }

def call_policy_llm(policy_text: str, book_payload: dict, model: str) -> dict:
    """
    调用大模型，根据 policy_rules.txt + 单本图书信息给出决策。
    约定输出 JSON，形如：
    {
      "decision_level": "采 / 建议采 / 不采 / 需人工复核",
      "policy_score": 85,
      "suggest_copies": 2,
      "rules_hit": ["学术价值较高", "与齐鲁文化相关"],
      "comment": "简短两句理由"
    }
    """
    user_content = (
        "下面是一本待选图书的信息（JSON）：\n"
        + json.dumps(book_payload, ensure_ascii=False)
        + "\n\n请严格根据上面的“图书采选政策”进行判断，并仅用 JSON 格式回答：\n"
        "{\n"
        '  "decision_level": "采/建议采/不采/需人工复核",\n'
        '  "policy_score": 整数0到100,\n'
        '  "suggest_copies": 0/1/2/3,\n'
        '  "rules_hit": ["命中的关键规则或宏观调控点1", "规则2"],\n'
        '  "comment": "不超过两句话的中文说明"\n'
        "}\n"
        "不要输出任何多余文字。"
    )
    messages = [
        {"role": "system", "content": (
                "你是山东师范大学图书馆的采访决策助手，需要严格按照以下采选政策进行判断。\n"
                "政策内容如下：\n"
                f"{policy_text}"
            )
         },
        {"role": "user", "content": user_content}
    ]

    try:
        # resp = client.chat.completions.create(
        #     model=model,
        #     messages=messages,
        #     temperature=0.0,
        # )
        resp= SLLM.call_llm_P(messages)
        # typeRes = resp.content
        # typeMess = messages[1]
        # print({"call_llm原文": f"{typeMess}"})
        # print({"call_llm": f"问题结果：{typeRes}"})

        content = resp.content.strip()
        # 简单防守：如果外面包了代码块，去掉 ```json
        content = content.strip("` \n")
        print(f"2222222{content}")

        return content
        # return json.loads(content)
    except Exception as e:
        print("调用 policy LLM 失败：", e)
        # 出错时返回一个保守结果，标记需人工复核
        return {
            "decision_level": "需人工复核",
            "policy_score": 50,
            "suggest_copies": 0,
            "rules_hit": ["LLM调用失败"],
            "comment": "LLM 调用失败或输出解析错误，建议人工复核。"
        }

def apply_policy_llm_to_df(
    df_scored: pd.DataFrame,
    policy_text: str,
    top_k: int = 200,
    model: str = "deepSeek",
) -> pd.DataFrame:
    """
    对 df_scored 中的一部分书（默认：rule_hard_reject=0 且按 score_final 排名前 top_k）
    调用 LLM 做政策评估，在 df 中新增以下列：
    - policy_decision_level
    - policy_score
    - policy_suggest_copies
    - policy_rules_hit
    - policy_comment
    """
    df = df_scored.copy()

    # 初始化列
    df["policy_decision_level"] = None
    df["policy_score"] = np.nan
    df["policy_suggest_copies"] = np.nan
    df["policy_rules_hit"] = None
    df["policy_comment"] = None

    # 选出要送给 LLM 的候选集合
    candidates = df[df["rule_hard_reject"] == 0].sort_values(
        "score_final", ascending=False
    ).head(top_k)

    for idx, row in candidates.iterrows():
        payload = build_book_policy_payload(row)
        result = call_policy_llm(policy_text, payload, model=model)

        content = result.content.strip() if hasattr(result, 'content') else str(result)

        # 关键：如果内容为空，抛出自定义异常
        if not content:
            raise ValueError("LLM返回空响应，请检查API调用")

        # 清理前缀
        json_start = content.find('{')
        if json_start == -1:
            raise ValueError(f"未找到JSON数据: {content}")

        json_str = content[json_start:]
        result=json.loads(json_str)

        df.at[idx, "policy_decision_level"] = result.get("decision_level")
        df.at[idx, "policy_score"] = result.get("policy_score")
        df.at[idx, "policy_suggest_copies"] = result.get("suggest_copies")
        df.at[idx, "policy_rules_hit"] = "; ".join(result.get("rules_hit", []))
        df.at[idx, "policy_comment"] = result.get("comment")

    return df

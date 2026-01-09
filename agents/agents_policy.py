#Step2 政策审查智能体
from typing import Optional
import re
import json
import pandas as pd
import Single_Agent
from agents.text_utils import build_book_text
from agents.io_utils import safe_float

HUMAN_COLS = ["_row_id", "人工判定", "人工一级分类", "人工备注", "人工覆盖", "人工需复核"]
POLICY_AUDIT_COLS = ["政策原始输出", "政策解析成功"]

def load_policy_text(policy_path: str) -> str:
    with open(policy_path, "r", encoding="utf-8") as f:
        return f.read()

def policy_check_llm(policy_text: str, book_text: str, category: str) -> dict:
    system_prompt = f"""你是山东师范大学图书馆“采选政策审查官”。
【政策全文】
{policy_text}
【政策全文结束】
要求：只输出 JSON，不要输出任何多余文字，不要使用 ```json 代码块。"""

    user_prompt = f"""候选图书信息：
【学科大类】{category}
【图书信息】
{book_text}

只输出严格 JSON：
{{
  "policy_decision": "采/建议采/不采/需人工复核",
  "policy_score": 0-100的整数,
  "policy_copies_cap": 0-3的整数,
  "policy_rules_hit": ["命中的关键条款摘要1","条款2"],
  "policy_reason": "不超过两句话说明"
}}"""

    raw1 = Single_Agent.call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    obj = parse_llm_json_safely(raw1)
    if obj is not None:
        obj["_policy_raw"] = str(raw1)[:5000]  # 记录原始输出，最多存5000字
        obj["_policy_parse_ok"] = 1
        return obj

    # 失败重试：更强硬地要求“只给JSON”
    retry_prompt = f"""你刚才的输出不是合法 JSON（无法被程序解析）。
请你现在【只输出一段合法 JSON】（不要代码块、不要解释文字），JSON 结构必须完全一致。"""
    raw2 = Single_Agent.call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(raw1)},
        {"role": "user", "content": retry_prompt},
    ])

    obj2 = parse_llm_json_safely(raw2)
    if obj2 is not None:
        obj2["_policy_raw"] = str(raw2)[:5000]
        obj2["_policy_parse_ok"] = 1
        return obj2

    # 仍失败：兜底
    return {
        "policy_decision": "需人工复核",
        "policy_score": 50,
        "policy_copies_cap": 1,
        "policy_rules_hit": ["JSON解析失败"],
        "policy_reason": "模型输出未能解析为JSON，建议人工复核。",
        "_policy_raw": str(raw2)[:5000],
        "_policy_parse_ok": 0,
    }

def run_policy_agent(
    df: pd.DataFrame,
    policy_text: str,
    out_path: str,
    scope: str = "accepted_only",
    top_k: Optional[int] = None,
    rank_cols=("score_final_adj", "score_final", "LLM_推荐分数", "score_model"),
) -> pd.DataFrame:
    df = df.copy()

    # --- 确保人工列/审计列存在（不影响已有值，防止后续模块误以为“缺列”）---
    for c in HUMAN_COLS:
        if c not in df.columns:
            df[c] = "" if c not in ("人工覆盖", "人工需复核") else 0

    for c in POLICY_AUDIT_COLS:
        if c not in df.columns:
            df[c] = "" if c == "政策原始输出" else 0

    for c in ["政策决策","政策得分","政策册数上限","政策命中条款","政策理由","政策放行"]:
        if c not in df.columns:
            df[c] = 0 if c in ("政策得分","政策册数上限","政策放行") else ""

    def rank_score(row) -> float:
        for c in rank_cols:
            if c in row.index and pd.notna(row[c]):
                return safe_float(row[c], 0.0)
        return 0.0



    # ✅ 0) 人工直推：跳过 LLM 政策审查，直接放行并写明原因
    mask_force = (df.get("人工直推", 0) == 1)
    if mask_force.any():
        df.loc[mask_force, "政策决策"] = "人工直推"
        df.loc[mask_force, "政策得分"] = 100
        df.loc[mask_force, "政策册数上限"] = 3
        df.loc[mask_force, "政策命中条款"] = "人工反馈：接受 => 直推"
        df.loc[mask_force, "政策理由"] = "人工判定为接受，按规则跳过政策模型审查并放行。"
        df.loc[mask_force, "政策放行"] = 1

    work = df[~mask_force].copy()

    if scope == "accepted_only" and "采纳建议" in df.columns:
        work = work[work["采纳建议"] == "接受"]

    if top_k is not None:
        tmp = work.copy()
        tmp["_rank"] = tmp.apply(rank_score, axis=1)
        work = tmp.sort_values("_rank", ascending=False).head(top_k)

    total = len(work)
    print(f"[Agent2-政策] 审查范围记录数：{total}")

    for i, (idx, row) in enumerate(work.iterrows(), start=1):
        book_text = build_book_text(row)
        cat = str(row.get("一级分类", "其他类")) or "其他类"

        result = policy_check_llm(policy_text, book_text, cat)

        decision = result.get("policy_decision", "需人工复核")
        score = int(result.get("policy_score", 50))
        cap = int(result.get("policy_copies_cap", 1))
        rules_hit = "; ".join(result.get("policy_rules_hit", []))
        reason = result.get("policy_reason", "")

        df.at[idx, "政策决策"] = decision
        df.at[idx, "政策得分"] = score
        df.at[idx, "政策册数上限"] = cap
        df.at[idx, "政策命中条款"] = rules_hit
        df.at[idx, "政策理由"] = reason
        df.at[idx, "政策放行"] = 1 if decision in ("采","建议采") else 0

        df.at[idx, "政策原始输出"] = result.get("_policy_raw", "")
        df.at[idx, "政策解析成功"] = int(result.get("_policy_parse_ok", 0))



        if i % 20 == 0 or i == total:
            print(f"\r[Agent2-政策] {i}/{total}", end="", flush=True)
    print()

    # 在 run_policy_agent() 处理完 work 之后追加：
    mask_missing = df["政策决策"].isna() | (df["政策决策"].astype(str).str.strip() == "")

    df.loc[mask_missing, "政策决策"] = "待审查"
    df.loc[mask_missing, "政策得分"] = 0
    df.loc[mask_missing, "政策册数上限"] = 1
    df.loc[mask_missing, "政策命中条款"] = ""
    df.loc[mask_missing, "政策理由"] = "本轮未进行政策审查（可能因 top_k / 范围限制）。"

    # 策略：未审查也允许进入“临时评判”，前提是学科接受
    # df.loc[mask_missing, "政策放行"] = (df["采纳建议"] == "接受").astype(int)
    if "采纳建议" in df.columns:
        df.loc[mask_missing, "政策放行"] = (df.loc[mask_missing, "采纳建议"] == "接受").astype(int)
    else:
        df.loc[mask_missing, "政策放行"] = 0

    # 同时把审计列也补齐（让待审查条目可追踪）
    df.loc[mask_missing, "政策原始输出"] = df.loc[mask_missing, "政策原始输出"].fillna("")
    df.loc[mask_missing, "政策解析成功"] = df.loc[mask_missing, "政策解析成功"].fillna(0).astype(int)

    df.to_excel(out_path, index=False)
    print(f"[Agent2-政策] 输出：{out_path}")
    return df

def parse_llm_json_safely(raw: str) -> Optional[dict]:
    """
    尽最大可能从 LLM 输出中提取 JSON 对象并解析。
    兼容 ```json 代码块、前后解释文字、中文引号、末尾逗号等。
    """
    if raw is None:
        return None
    text = str(raw).strip()

    # 去掉代码块围栏
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    # 提取第一个 {...}（跨行）
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    j = m.group(0).strip()

    # 清理常见非标准符号
    j = j.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    # 去掉末尾逗号：{"a":1,}
    j = re.sub(r",\s*([}\]])", r"\1", j)

    try:
        return json.loads(j)
    except Exception:
        return None
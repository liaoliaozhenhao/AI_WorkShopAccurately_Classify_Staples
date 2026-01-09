# 学科小样本测试
import os
import pandas as pd
import re
from typing import Tuple, Optional, Dict, Any

def ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["_row_id"] = df.index
    return df

def sample_for_calibration(df: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    抽10-30条小样本：优先覆盖多样性（简单版本：随机 + 若有分数列则分位抽样）
    """
    df = ensure_row_id(df)
    n = max(10, min(int(n), 30))

    # 若已有某种分数列，按高/中/低分位抽样更有“校准价值”
    score_cols = [c for c in ["score_final_adj", "score_final", "score_model", "LLM_推荐分数"] if c in df.columns]
    if score_cols:
        s = df[score_cols[0]].astype(float).fillna(0.0)
        df2 = df.assign(_s=s)

        hi = df2.nlargest(n // 3, "_s")
        lo = df2.nsmallest(n // 3, "_s")
        mid = df2.sample(n=n - len(hi) - len(lo), random_state=seed)
        out = pd.concat([hi, mid, lo], axis=0).drop_duplicates("_row_id").head(n)
        return out.drop(columns=["_s"], errors="ignore")

    return df.sample(n=n, random_state=seed)

def export_review_sheet(sample_df: pd.DataFrame, out_path: str) -> str:
    """
    输出人工反馈表模板：保留关键字段 + 留空人工列
    """
    sample_df = sample_df.copy()

    for c in ["人工判定", "人工一级分类", "人工备注"]:
        if c not in sample_df.columns:
            sample_df[c] = ""

    # 把人工列放前面，便于填写
    front = ["_row_id", "人工判定", "人工一级分类", "人工备注"]
    cols = front + [c for c in sample_df.columns if c not in front]
    sample_df = sample_df[cols]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sample_df.to_excel(out_path, index=False)
    return out_path

def load_human_feedback(review_path: str) -> pd.DataFrame:
    df = pd.read_excel(review_path)
    df = ensure_row_id(df)
    # 只保留填写了人工判定的行
    df["人工判定"] = df["人工判定"].astype(str).str.strip()
    return df[df["人工判定"].isin(["接受", "拒绝", "复核"])].copy()

def build_feedback_notes(
    feedback_df: pd.DataFrame,
    mode: str = "soft",          # soft / off / full
    min_samples: int = 12,       # 少于该值就不启用（只写弱提示）
    max_items: int = 8,          # 最多保留多少条典型备注
) -> str:
    """
    生成“纠偏要点”（弱提示版，避免小样本过拟合）：
    - mode="off": 不启用纠偏（会覆盖写入一个“本轮不启用纠偏”的文件，避免旧文件继续影响）
    - mode="soft": 只输出原则+样本统计+少量典型备注（不输出关键词Top榜）
    - mode="full": 兼容你原来版本（如确实需要关键词榜时再开）
    """
    n = len(feedback_df)
    lines = []
    lines.append("【近期人工反馈纠偏要点】（用于学科智能体本轮轻量校准）")
    lines.append("【人工反馈使用原则（非常重要）】")
    lines.append("- 以下反馈仅作“轻微调整倾向”的参考，不得推翻：学科基本判别、政策硬规则、馆藏画像配额逻辑。")
    lines.append("- 当反馈与书目信息明显不一致时，以书目信息为准。")
    lines.append("- 仅用于边界/不确定条目：优先输出“复核”，不要强行接受或拒绝。")
    lines.append("- 不要把备注中的关键词当作新的硬规则。")

    vc = feedback_df.get("人工判定", pd.Series([], dtype=str)).value_counts().to_dict()
    lines.append(f"【本次校准样本量】{n}（接受{vc.get('接受',0)} / 拒绝{vc.get('拒绝',0)} / 复核{vc.get('复核',0)}）")

    # ---- 门控：样本太少或 mode=off 时，不写“纠偏内容”，只写原则与统计 ----
    if mode == "off" or n < min_samples:
        lines.append("【本轮处理策略】本轮样本量不足或已设置关闭纠偏：仅保留使用原则，不注入具体纠偏要点（防止过拟合）。")
        return "\n".join(lines)

    # ---- soft：输出少量“典型备注”，不输出关键词Top榜 ----
    if mode == "soft":
        lines.append("【典型人工备注（仅供参考，最多保留若干条）】")
        # 只取非空备注
        remarks = feedback_df.get("人工备注", pd.Series([], dtype=str)).fillna("").astype(str)
        # 取去重后的前 max_items 条（或你也可以按“长度/信息量”排序）
        items = []
        for r in remarks:
            rr = re.sub(r"\s+", " ", r).strip()
            if rr and rr not in items:
                items.append(rr)
            if len(items) >= max_items:
                break
        if not items:
            lines.append("- （无有效人工备注）")
        else:
            for r in items:
                lines.append(f"- {r[:200]}")
        lines.append("【提醒】以上为“弱提示”，只用于边界条目；对明显不相关的书目应忽略。")
        return "\n".join(lines)

    # ---- full：如果你未来需要再加回“关键词榜”，可以在这里写你原来的逻辑 ----
    # 为了安全，full 也建议缩小 top 数量，且加“弱提示”前缀
    lines.append("【full 模式未开启关键词榜】建议默认使用 soft，避免过拟合。")
    return "\n".join(lines)

def re_split_multi(text: str):
    import re
    return re.split(r"[，,;；、\s]+", text)

def write_feedback_file(feedback_notes: str, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(feedback_notes.strip() + "\n")
    return out_path


def _short(s: str, n: int = 80) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]

def _extract_keywords(text: str, k: int = 6) -> str:
    """非常轻量的关键词：从题名/简介里抓几个长一点的词片段（不依赖外部库）"""
    if not text:
        return ""
    t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", str(text))
    toks = [x.strip() for x in t.split() if len(x.strip()) >= 2]
    # 去重保序
    seen, out = set(), []
    for w in toks:
        if w not in seen:
            out.append(w)
            seen.add(w)
        if len(out) >= k:
            break
    return "、".join(out)

def auto_remark_from_row(row: pd.Series) -> str:
    """根据行内信息自动生成备注（当人工备注为空时使用）"""
    title = str(row.get("题名", "") or row.get("正题名", "") or "")
    intro = str(row.get("内容简介", "") or "")
    reason = str(row.get("推荐理由", "") or "")

    kw = _extract_keywords(title + " " + _short(intro, 120), k=6)
    parts = []
    if kw:
        parts.append(f"关键词：{kw}")
    if reason.strip():
        parts.append(f"模型理由：{_short(reason, 120)}")
    if not parts:
        parts.append("自动备注：未提供足够文本字段（题名/简介/理由）")

    return "；".join(parts)

def auto_fill_feedback_remarks(review_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    自动补备注：
    - 对“人工判定”有效但“人工备注”为空的行，自动生成备注
    - 可选择另存为新文件（推荐）
    """
    df = pd.read_excel(review_path, engine='openpyxl')
    if "_row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["_row_id"] = df.index

    if "人工备注" not in df.columns:
        df["人工备注"] = ""

    # 只对填了人工判定的行补备注
    df["人工判定"] = df.get("人工判定", "").astype(str).str.strip()
    mask_valid = df["人工判定"].isin(["接受", "拒绝", "复核"])
    mask_empty_remark = df["人工备注"].isna() | (df["人工备注"].astype(str).str.strip() == "")
    mask = mask_valid & mask_empty_remark

    if mask.any():
        df.loc[mask, "人工备注"] = df.loc[mask].apply(auto_remark_from_row, axis=1)

    if save_path:
        df.to_excel(save_path, index=False)
    return df
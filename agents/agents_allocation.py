#Step3 馆藏画像配种 + 配册

import json
import pandas as pd
import re
from typing import Dict, Any, Tuple, Optional

from agents.io_utils import safe_float


def load_profile(profile_path: str) -> dict:
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_call_prefix(class_no) -> str:
    if pd.isna(class_no):
        return ""
    s = str(class_no).strip().upper()
    m = re.match(r"([A-Z]{1,2})", s)
    return m.group(1) if m else ""

def build_category_weights(profile: dict) -> dict:
    ds = profile.get("demand_supply", [])
    raw = {}
    for it in ds:
        cat = it.get("category")
        if not cat:
            continue
        raw[cat] = float(it.get("借阅次数/馆藏复本", 0.0))
    s = sum(raw.values()) or 1.0
    return {k: v / s for k, v in raw.items()}

def _norm_hot_map(items, key_field: str, borrow_field="借书总次数", copies_field="复本总数"):
    scores = {}
    vals = []
    for it in items:
        k = it.get(key_field)
        if not k:
            continue
        b = float(it.get(borrow_field, 0.0))
        c = float(it.get(copies_field, 1.0)) if float(it.get(copies_field, 1.0)) > 0 else 1.0
        v = b / c
        scores[str(k)] = v
        vals.append(v)
    if not vals:
        return {}
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}

def compute_rank_score(row, rank_cols=("score_final_adj","score_final","LLM_推荐分数","score_model","政策得分")) -> float:
    # 把各种分数做一个“可比较”的排序分（只是排序用）
    for c in rank_cols:
        if c in row.index and pd.notna(row[c]):
            return safe_float(row[c], 0.0)
    return 0.0

def allocate_kinds_and_copies(
    df: pd.DataFrame,
    profile: dict,
    out_path: str,
    summary_path: str,
    total_kind_budget: Optional[int] = None,   # 总种数预算（None=不裁剪，全部放行但仍给出建议配额）
    per_category_min: int = 0,
    per_category_max: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    force = (df.get("人工直推", 0) == 1)

    # 先初始化
    df["final_selected"] = df.get("final_selected", 0)
    df["推荐册数"] = df.get("推荐册数", 0)

    # 人工直推直接入选
    df.loc[force, "final_selected"] = 1
    df.loc[force & (df["推荐册数"] <= 0), "推荐册数"] = 1


    # 规范类别名称：你的 Agent1 输出应是 “其他类”
    if "一级分类" not in df.columns:
        df["一级分类"] = "其他类"

    df["call_prefix"] = df["分类号"].apply(infer_call_prefix) if "分类号" in df.columns else ""
    df["rank_score"] = df.apply(compute_rank_score, axis=1)

    cat_weights = build_category_weights(profile)
    borrow_profile = profile.get("borrow_profile", {})
    prefix_hot = _norm_hot_map(borrow_profile.get("top_call_prefix", []), "call_prefix")
    pub_hot = _norm_hot_map(borrow_profile.get("top_publishers", []), "出版社")

    # 配额池：学科接受 + 政策放行
    force = (df.get("人工直推", 0) == 1)

    ok_subject = (df["采纳建议"] == "接受") if "采纳建议" in df.columns else True
    ok_policy = (df["政策放行"] == 1) if "政策放行" in df.columns else df.get("政策决策", "").isin(["采", "建议采"])

    # ✅ 人工直推强行进入池
    pool = df[(ok_subject & ok_policy) | force].copy()

    # 画像需求分（0-1）：大类权重 + 分类前缀热度 + 出版社热度
    def profile_need_score(r):
        cat = r.get("一级分类", "其他类")
        cw = float(cat_weights.get(cat, 0.0))   # sum=1
        p = float(prefix_hot.get(r.get("call_prefix",""), 0.0))
        pub = str(r.get("出版社","") or "")
        ph = float(pub_hot.get(pub, 0.0))
        # 组合权重可调
        return max(0.0, min(1.0, 0.6 * (cw * 6.0) + 0.2 * p + 0.2 * ph))

    pool["画像需求分"] = pool.apply(profile_need_score, axis=1)
    df["画像需求分"] = 0.0
    df.loc[pool.index, "画像需求分"] = pool["画像需求分"]

    # 计算各类“建议配额（种数）”
    total_pool = len(pool)
    if total_kind_budget is None:
        total_kind_budget = total_pool

    quota = {}
    for cat, w in cat_weights.items():
        q = int(round(total_kind_budget * w))
        q = max(per_category_min, q)
        if per_category_max is not None:
            q = min(per_category_max, q)
        quota[cat] = q

    # 微调：让配额和等于 total_kind_budget
    diff = total_kind_budget - sum(quota.values())
    if quota and diff != 0:
        top_cat = max(cat_weights.items(), key=lambda x: x[1])[0]
        quota[top_cat] = max(0, quota.get(top_cat, 0) + diff)

    # 按类选 Top
    pool["final_selected"] = 0
    selected_list = []
    for cat, q in quota.items():
        sub = pool[pool["一级分类"] == cat].sort_values("rank_score", ascending=False)
        take = sub.head(q) if q > 0 else sub.iloc[0:0]
        pool.loc[take.index, "final_selected"] = 1
        selected_list.append(take)

    selected = pd.concat(selected_list, axis=0) if selected_list else pool.iloc[0:0]

    # 推荐册数：画像需求分 + 政策册数上限
    def recommend_copies(r):
        if int(r.get("final_selected", 0)) != 1:
            return 0
        need = safe_float(r.get("画像需求分", 0.0), 0.0)
        cap = int(safe_float(r.get("政策册数上限", 3), 3))

        base = 1
        if need >= 0.75:
            base = 3
        elif need >= 0.55:
            base = 2
        else:
            base = 1
        base = min(base, cap)
        return max(1, base)

    selected["推荐册数"] = selected.apply(recommend_copies, axis=1)

    # ✅ 不要清零整表（会覆盖人工直推/已选结果）
    if "final_selected" not in df.columns:
        df["final_selected"] = 0
    if "推荐册数" not in df.columns:
        df["推荐册数"] = 0

    df.loc[selected.index, "final_selected"] = 1
    df.loc[selected.index, "推荐册数"] = selected["推荐册数"]

    # ✅ 人工直推兜底：确保入选且至少 1 册
    df.loc[force, "final_selected"] = 1
    df.loc[force & (df["推荐册数"] <= 0), "推荐册数"] = 1

    # 给全表增加“建议配额”列（便于人工看配种结构）
    df["建议配额_该类种数"] = df["一级分类"].map(lambda c: quota.get(c, 0))

    # 汇总
    summary = (
        df[df["final_selected"] == 1]
        .groupby("一级分类")
        .agg(
            最终入选种数=("final_selected", "sum"),
            推荐册数总计=("推荐册数", "sum"),
            平均画像分=("画像需求分", "mean"),
            平均rank分=("rank_score", "mean"),
        )
        .reset_index()
    )
    summary["建议配额_种数"] = summary["一级分类"].map(lambda c: quota.get(c, 0))

    # 写回 selected 后，立刻恢复人工直推（避免被 quota/重置覆盖）
    df.loc[force, "final_selected"] = 1

    mask_force_copies = force & (df["推荐册数"] <= 0)
    df.loc[mask_force_copies, "推荐册数"] = 1

    df.to_excel(out_path, index=False)
    summary.to_excel(summary_path, index=False)
    print(f"[Agent3-画像] 输出：{out_path}")
    print(f"[Agent3-画像] 汇总：{summary_path}")

    # allocation 阶段最后兜底：入选但册数为0 -> 至少1册
    mask = (df["final_selected"] == 1) & (df["推荐册数"] <= 0)
    df.loc[mask, "推荐册数"] = 1

    return df, summary

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
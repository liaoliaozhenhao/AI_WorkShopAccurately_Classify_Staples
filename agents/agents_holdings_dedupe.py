# 馆藏重复清单 筛查

import re
import pandas as pd

def _norm(s: str) -> str:
    """
    归一化文本：
    - None/NaN -> ""
    - 去首尾空格 -> 去所有空白
    - 去除标点符号
    - 转小写
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    # 去掉除“字母数字下划线中文”以外的字符（标点符号等）
    s = re.sub(r"[^\w\u4e00-\u9fa5]+", "", s)
    return s


def _pick(row, *cands) -> str:
    """
    从行里按候选列名依次取第一个非空值
    """
    for c in cands:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            return str(row[c]).strip()
    return ""


# def build_holdings_index(df_holdings: pd.DataFrame) -> dict:
#     """
#     用【题名+责任者】构建馆藏索引集合：
#       {"title_author": set("归一化题名|归一化责任者", ...)}
#     """
#     keys = set()
#
#     for _, r in df_holdings.iterrows():
#         title = _norm(_pick(r, "题名", "正题名", "书名", "题名与责任者"))
#         author = _norm(_pick(r, "责任者", "作者", "著者", "主编", "编著", "编"))
#
#         if title and author:
#             keys.add(f"{title}|{author}")
#         # 如果馆藏表责任者缺失，你也可以选择“只题名”兜底：
#         # elif title:
#         #     keys.add(f"{title}|")
#     return {"title_author": keys}
#
#
# def mark_and_split_dedupe(df_raw: pd.DataFrame, holdings_index: dict):
#     """
#     按【题名+责任者】严格匹配拆分数据：
#     - df_dup：馆藏重复（题名+责任者命中）
#     - df_eval：可进入后续评判的征订条目
#     """
#     df = df_raw.copy()
#
#     if "馆藏重复" not in df.columns:
#         df["馆藏重复"] = 0
#     if "馆藏重复依据" not in df.columns:
#         df["馆藏重复依据"] = ""
#
#     key_set = holdings_index.get("title_author", set())
#     dup_idx = []
#
#     for idx, r in df.iterrows():
#         title = _norm(_pick(r, "题名", "正题名", "书名", "题名与责任者"))
#         author = _norm(_pick(r, "责任者", "作者", "著者", "主编", "编著", "编"))
#
#         if title and author:
#             key = f"{title}|{author}"
#             if key in key_set:
#                 df.at[idx, "馆藏重复"] = 1
#                 df.at[idx, "馆藏重复依据"] = "题名+责任者命中"
#                 dup_idx.append(idx)
#
#     df_dup = df.loc[dup_idx].copy() if dup_idx else df.iloc[0:0].copy()
#     df_eval = df.drop(index=dup_idx).copy() if dup_idx else df.copy()
#     return df_eval, df_dup

def build_holdings_index(df_holdings: pd.DataFrame) -> dict:
    """
    仅用题名构建馆藏索引集合：
    返回：
      {"title": set(...)}  # 归一化后的题名集合
    """
    titles = set()
    for _, r in df_holdings.iterrows():
        title = _norm(_pick(r, "题名", "正题名", "书名", "题名与责任者"))
        if title:
            titles.add(title)
    return {"title": titles}


def mark_and_split_dedupe(df_raw: pd.DataFrame, holdings_index: dict):
    """
    仅按题名严格匹配拆分数据：
    - df_dup：馆藏重复（题名命中）
    - df_eval：可进入后续评判的征订条目
    """
    df = df_raw.copy()

    if "馆藏重复" not in df.columns:
        df["馆藏重复"] = 0
    if "馆藏重复依据" not in df.columns:
        df["馆藏重复依据"] = ""

    holdings_titles = holdings_index.get("title", set())
    dup_idx = []

    for idx, r in df.iterrows():
        title = _norm(_pick(r, "题名", "正题名", "书名", "题名与责任者"))
        if title and title in holdings_titles:
            df.at[idx, "馆藏重复"] = 1
            df.at[idx, "馆藏重复依据"] = "题名命中"
            dup_idx.append(idx)

    df_dup = df.loc[dup_idx].copy() if dup_idx else df.iloc[0:0].copy()
    df_eval = df.drop(index=dup_idx).copy() if dup_idx else df.copy()
    print(f"馆藏重复（题名命中）数量{len(df_dup)},可进入后续评判的征订条目数量{len(df_eval)}")
    return df_eval, df_dup
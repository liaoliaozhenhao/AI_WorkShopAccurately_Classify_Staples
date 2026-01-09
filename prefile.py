#读取画像报告 & 预处理

import os
import re
import math
import pickle
import json  # 新增
from typing import Union, Optional

import numpy as np
import pandas as pd
def load_collection_profile(path: str):
    """
    读取《馆藏与借阅画像分析报告 (1).xlsx》，
    返回若干 DataFrame 组成的画像字典。
    """
    xls = pd.ExcelFile(path)

    # 1) 学科供需对比
    df_subject = pd.read_excel(xls, sheet_name="供需对比_学科")
    # 借阅次数/馆藏复本 已经是一个“需求强度”，再做 0-1 归一化
    def min_max_norm(s: pd.Series):
        s = s.astype(float)
        minv, maxv = s.min(), s.max()
        if maxv == minv:
            return pd.Series(1.0, index=s.index)
        return (s - minv) / (maxv - minv)

    df_subject["subject_need_score"] = min_max_norm(df_subject["借阅次数/馆藏复本"])

    # 2) 分类前缀借阅画像
    df_b_prefix = pd.read_excel(xls, sheet_name="借阅_分类前缀Top300")
    df_b_prefix["borrow_intensity"] = (
        df_b_prefix["借书总次数"] / df_b_prefix["复本总数"].clip(lower=1)
    )
    df_b_prefix["prefix_need_score"] = min_max_norm(df_b_prefix["borrow_intensity"])

    # 3) 馆藏分类前缀（这里暂时只做对照，不算分）
    df_h_prefix = pd.read_excel(xls, sheet_name="馆藏_分类前缀Top300")

    # 4) 出版社借阅画像
    df_b_pub = pd.read_excel(xls, sheet_name="借阅_出版社Top300")
    df_b_pub["borrow_intensity"] = (
        df_b_pub["借书总次数"] / df_b_pub["复本总数"].clip(lower=1)
    )
    df_b_pub["pub_need_score"] = min_max_norm(df_b_pub["borrow_intensity"])

    # 5) 馆藏出版社
    df_h_pub = pd.read_excel(xls, sheet_name="馆藏_出版社Top300")

    profile = {
        "subject": df_subject.set_index("category"),
        "borrow_prefix": df_b_prefix.set_index("call_prefix"),
        "hold_prefix": df_h_prefix.set_index("call_prefix"),
        "borrow_pub": df_b_pub.set_index("出版社"),
        "hold_pub": df_h_pub.set_index("出版社"),
    }
    return profile


def infer_call_prefix_from_classno(class_no: Union[str, float]) -> Optional[str]:
    """从分类号中抽取前缀，如 I, TP, TS 等。"""
    if pd.isna(class_no):
        return None
    s = str(class_no).strip().upper()
    m = re.match(r"([A-Z]{1,2})", s)
    return m.group(1) if m else None


def infer_subject_category_from_classno(class_no: Union[str, float]) -> str:
    """
    按中图法大类粗略映射到画像里的六个学科大类：
    人文科学类 / 社会科学类 / 理工类 / 语言教育类 / 艺术类 / 其他类
    """
    if pd.isna(class_no):
        return "其他类"
    s = str(class_no).strip().upper()
    if not s:
        return "其他类"
    first = s[0]

    # 语言教育类：以 H 为主
    if first == "H":
        return "语言教育类"

    # 艺术类：J
    if first == "J":
        return "艺术类"

    # 社会科学类：C,D,F
    if first in ("C", "D", "F"):
        return "社会科学类"

    # 人文科学类：A,B,G,I,K
    if first in ("A", "B", "G", "I", "K"):
        return "人文科学类"

    # 理工类：N,O,P,Q,R,S,T,U,V,X 等自然科学+工程
    if first in ("N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "E"):
        return "理工类"

    return "其他类"


def add_collection_need_scores(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    """
    根据馆藏&借阅画像，为每本书增加：
    - subject_category 学科大类
    - call_prefix 分类前缀
    - need_subject_score
    - need_prefix_score
    - need_publisher_score
    - need_overall_score (总需求分)
    """
    df = df.copy()

    # 推出学科大类 & 分类前缀
    df["subject_category"] = df["分类号"].apply(
        infer_subject_category_from_classno
    )
    df["call_prefix"] = df["分类号"].apply(
        infer_call_prefix_from_classno
    )

    # 1) 学科需求分
    subj_score_map = profile["subject"]["subject_need_score"].to_dict()

    def get_subj_score(cat: str) -> float:
        return float(subj_score_map.get(cat, 0.0))

    df["need_subject_score"] = df["subject_category"].apply(get_subj_score)

    # 2) 分类前缀需求分
    prefix_need_map = profile["borrow_prefix"]["prefix_need_score"].to_dict()

    def get_prefix_score(prefix: Optional[str]) -> float:
        if prefix is None:
            return 0.0
        return float(prefix_need_map.get(prefix, 0.0))

    df["need_prefix_score"] = df["call_prefix"].apply(get_prefix_score)

    # 3) 出版社需求分
    pub_need_map = profile["borrow_pub"]["pub_need_score"].to_dict()

    def get_pub_score(pub: Union[str, float]) -> float:
        if pd.isna(pub):
            return 0.0
        return float(pub_need_map.get(str(pub), 0.0))

    df["need_publisher_score"] = df["出版社"].apply(get_pub_score)

    # 4) 综合需求分：权重可以自己调
    df["need_overall_score"] = (
        0.5 * df["need_subject_score"]
        + 0.3 * df["need_prefix_score"]
        + 0.2 * df["need_publisher_score"]
    )

    return df

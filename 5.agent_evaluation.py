import os
import re
import math
import pickle

import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# 你的原有代码
# df.to_excel("模型选书结果_仅推理_征订单.xlsx", ...)
'''
# 单LLM 单分类推理结果评价
'''

# ======================
# 全局配置
# ======================


MODEL_PATH = "book_selector_online.pkl"  # 模型持久化路径
# OUT_PATH = "模型选书结果_模拟.xlsx"        # 输出结果



# 参数配置区
CONFIG = {
    "threshold": 0.5,  # 选书门槛（越高越严格）   0.5
    "pos_rule_weight": 0.3,  # 正向规则加分权重
    "neg_rule_weight": 0.1,  # 负向规则减分权重
    "class_weight": {0: 1, 1: 10},  # 训练时“选中”类别的权重

    "machine_weight":0.9,
    "LLM_weight":0.1,

    # "machine_weight":0.982111,
    # "LLM_weight":0.017889,


}
RECOMMEND_THRESHOLD = CONFIG["threshold"]                # score_final >= 该阈值则判定为“选中”
# CLASS_WEIGHT = {0: 1, 1: 10}                              # 正负样本不均衡时的类别权重


# ======================
# 一些通用小工具
# ======================

def normalize_isbn(x):
    """去掉 ISBN 中的横线等，便于匹配。"""
    if pd.isna(x):
        return ""
    return str(x).replace("-", "").strip()


def concat_text(row, cols):
    """把多列文本合并为一个字符串。"""
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]))
    return " ".join(parts)


def contains_any(text, keywords):
    """判断 text 中是否包含任意一个关键词。"""
    if not isinstance(text, str):
        return False
    return any(k in text for k in keywords)


def parse_size_cm(val):
    """从 '21cm' 这样的字符串中提取数值部分。"""
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r"(\d+(\.\d+)?)\s*cm", s)
    if m:
        return float(m.group(1))
    return np.nan


# ======================
# 数据读取与打标签
# ======================

def load_data(full_path: str, sel_path: str) -> pd.DataFrame:
    """读取征订目录和已订明细，返回带 is_selected 标签的总表。"""
    full = pd.read_excel(full_path)
    sel = pd.read_excel(sel_path, header=4)  # 你的订购明细前几行为说明，因此 header=4

    full["ISBN_norm"] = full["ISBN"].apply(normalize_isbn)
    sel["ISBN_norm"] = sel["ISBN"].apply(normalize_isbn)

    sel_isbns = set(sel["ISBN_norm"])
    full["is_selected"] = full["ISBN_norm"].isin(sel_isbns).astype(int)

    print("征订单总记录：", len(full))
    print("已订中记录数：", full["is_selected"].sum())
    return full


# ======================
# 规则函数（根据你的订书原则）
# ======================

# —— 硬拒绝规则（出现一个就绝对不要） ——


def rule_vocational(row):
    """高职、高专等职业教育图书不要。"""
    text = concat_text(row, ["题名", "内容简介", "备注"])
    return contains_any(text, ["高职", "高专", "职业学院", "职业教育", "中职", "技工", "中等职业"])


def rule_small_size(row):
    """开本 / 尺寸小于 19cm 不要（只有有相关字段才生效）。"""
    size_col = None
    if "尺寸" in row:
        size_col = row["尺寸"]
    elif "开本" in row:
        size_col = row["开本"]
    size_cm = parse_size_cm(size_col) if size_col is not None else np.nan
    return (not np.isnan(size_cm)) and (size_cm < 19)


def rule_few_pages(row, threshold=120):
    """页数太少的图书不要（例如 < 120 页）。"""
    pages = None
    if "页码" in row:
        pages = row["页码"]
    elif "页数" in row:
        pages = row["页数"]
    if pages is None or (isinstance(pages, float) and math.isnan(pages)):
        return False
    try:
        p = int(float(pages))
    except Exception:
        return False
    return p < threshold


def rule_cartoon(row):
    """连环画、漫画、绘本等不要。"""
    text = concat_text(row, ["题名", "内容简介", "备注"])
    kws = ["连环画", "漫画", "绘本", "图画书", "宣传画", "卡通"]
    return contains_any(text, kws)


def rule_exam(row):
    """资格考试类图书不要。"""
    text = concat_text(row, ["题名", "内容简介", "备注"])
    kws = [
        "考试", "考点", "真题", "题库", "押题", "冲刺", "模拟试题", "试题解析",
        "历年试题", "辅导", "精讲精练", "过关", "考研", "国考", "省考",
        "教师资格", "资格考试", "职业资格", "注册"
    ]
    return contains_any(text, kws)


def rule_gov_tourism(row):
    """旅游厅、政府等出版物不要。"""
    text = concat_text(row, ["题名", "内容简介", "出版社"])
    kws = [
        "旅游厅", "文化和旅游厅", "人民政府", "市政府", "县政府",
        "自治区人民政府", "州人民政府", "政务", "政府公报", "政协", "人大常委会"
    ]
    return contains_any(text, kws)


def rule_open_univ(row):
    """国家开放大学出版社基本不要。"""
    pub = str(row.get("出版社", "") or "")
    return "国家开放大学出版社" in pub


def rule_micro_course(row):
    """微课图书基本不要。"""
    text = concat_text(row, ["题名", "内容简介"])
    return ("微课" in text) or ("慕课" in text)


def rule_engineering(row):
    """施工、土木、水利工程类不要。"""
    text = concat_text(row, ["题名", "内容简介", "分类号"])
    kws = ["施工", "土木工程", "水利工程", "桥梁工程", "隧道工程", "公路工程", "岩土工程", "市政工程"]
    return contains_any(text, kws)


def rule_audio_video_pub(row):
    """电子音像出版社等一般不要。"""
    pub = str(row.get("出版社", "") or "")
    return ("电子音像" in pub) or ("音像出版社" in pub)


# —— 软惩罚 / 软加分规则（模型和最终打分会用到） ——


def rule_shandong_qilu(row):
    """山东省、齐鲁文化类应优先订。"""
    text = concat_text(row, ["题名", "内容简介", "出版社"])
    kws = ["山东", "齐鲁", "齐鲁文化"]
    return contains_any(text, kws)


def rule_local_history(row):
    """乡村、县、地方志等优先订。"""
    text = concat_text(row, ["题名", "内容简介"])
    kws = ["地方志", "市志", "县志", "区志", "乡镇志", "乡志", "村志", "街道志", "县情", "市情"]
    return contains_any(text, kws)


def rule_expensive_history(row, price_threshold=200):
    """很贵的史料（K类且价格高）——只订一本，但属于优先关注。"""
    try:
        price = float(row.get("价格", np.nan))
    except Exception:
        price = np.nan
    class_no = str(row.get("分类号", "") or "")
    is_history = class_no.startswith("K")
    return is_history and (not np.isnan(price)) and price > price_threshold


def rule_econ_soft(row):
    """经济类尽量不要（F开头）。"""
    class_no = str(row.get("分类号", "") or "")
    return class_no.startswith("F")


def rule_law_plain(row):
    """一般法律类（非法律史/理论）少订。"""
    class_no = str(row.get("分类号", "") or "")
    is_law = class_no.startswith("D9") or class_no.startswith("D92") or class_no.startswith("D93")
    text = concat_text(row, ["题名", "内容简介"])
    good_kws = ["法律史", "法制史", "法学理论", "法律理论", "法哲学"]
    if is_law and not contains_any(text, good_kws):
        return True
    return False


def rule_marx_good(row):
    """马克思类图书，优先优质出版社。"""
    text = concat_text(row, ["题名", "内容简介"])
    pub = str(row.get("出版社", "") or "")
    has_marx = ("马克思" in text) or ("马列" in text) or ("马克思主义" in text)
    good_pubs = ["人民出版社", "高等教育出版社", "中国人民大学出版社", "商务印书馆", "中国社会科学出版社"]
    return has_marx and any(gp in pub for gp in good_pubs)


def rule_folk_bigcity(row):
    """大城市民俗类（上海等）可订。"""
    text = concat_text(row, ["题名", "内容简介"])
    if "民俗" not in text:
        return False
    big_cities = ["上海", "北京", "广州", "深圳", "南京", "杭州", "成都", "重庆", "天津", "武汉", "西安"]
    return contains_any(text, big_cities)


def rule_edu_sport_quality(row):
    """教育、体育类选择优质出版社。"""
    text = concat_text(row, ["题名", "内容简介"])
    has_edu_sport = ("教育" in text) or ("体育" in text)
    pub = str(row.get("出版社", "") or "")
    good_pubs = [
        "高等教育出版社", "人民教育出版社",
        "北京体育大学出版社", "上海体育学院出版社",
        "北京师范大学出版社"
    ]
    return has_edu_sport and any(gp in pub for gp in good_pubs)


def rule_green_architecture(row):
    """建筑类图书中，可持续、绿色理论的可以选。"""
    text = concat_text(row, ["题名", "内容简介"])
    has_arch = "建筑" in text
    kws = ["绿色建筑", "绿色", "可持续", "低碳", "生态", "节能", "环境友好"]
    return has_arch and contains_any(text, kws)


# 将规则统一应用到 DataFrame 上
def apply_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """对全表应用全部规则，返回带规则列的 DataFrame。"""
    df = df.copy()

    # 硬规则
    df["r_vocational"] = df.apply(rule_vocational, axis=1).astype(int)
    df["r_small_size"] = df.apply(rule_small_size, axis=1).astype(int)
    df["r_few_pages"] = df.apply(rule_few_pages, axis=1).astype(int)
    df["r_cartoon"] = df.apply(rule_cartoon, axis=1).astype(int)
    df["r_exam"] = df.apply(rule_exam, axis=1).astype(int)
    df["r_gov_tourism"] = df.apply(rule_gov_tourism, axis=1).astype(int)
    df["r_open_univ"] = df.apply(rule_open_univ, axis=1).astype(int)
    df["r_micro_course"] = df.apply(rule_micro_course, axis=1).astype(int)
    df["r_engineering"] = df.apply(rule_engineering, axis=1).astype(int)
    df["r_audio_video_pub"] = df.apply(rule_audio_video_pub, axis=1).astype(int)

    # 软规则
    df["r_shandong_qilu"] = df.apply(rule_shandong_qilu, axis=1).astype(int)
    df["r_local_history"] = df.apply(rule_local_history, axis=1).astype(int)
    df["r_expensive_history"] = df.apply(rule_expensive_history, axis=1).astype(int)
    df["r_econ_soft"] = df.apply(rule_econ_soft, axis=1).astype(int)
    df["r_law_plain"] = df.apply(rule_law_plain, axis=1).astype(int)
    df["r_marx_good"] = df.apply(rule_marx_good, axis=1).astype(int)
    df["r_folk_bigcity"] = df.apply(rule_folk_bigcity, axis=1).astype(int)
    df["r_edu_sport_quality"] = df.apply(rule_edu_sport_quality, axis=1).astype(int)
    df["r_green_architecture"] = df.apply(rule_green_architecture, axis=1).astype(int)

    # 汇总
    hard_cols = [
        "r_vocational", "r_small_size", "r_few_pages", "r_cartoon",
        "r_gov_tourism", "r_open_univ", "r_micro_course",
        "r_engineering", "r_exam"
    ]
    df["rule_hard_reject"] = (df[hard_cols].sum(axis=1) > 0).astype(int)

    pos_cols = [
        "r_shandong_qilu", "r_local_history", "r_marx_good",
        "r_folk_bigcity", "r_edu_sport_quality", "r_green_architecture"
    ]
    neg_soft_cols = ["r_econ_soft", "r_law_plain", "r_audio_video_pub"]

    df["rule_pos_score"] = df[pos_cols].sum(axis=1)
    df["rule_softneg_score"] = df[neg_soft_cols].sum(axis=1)

    return df


# ======================
# 文本特征与向量化
# ======================

TEXT_COLS = ["题名", "责任者", "出版社", "丛书题名", "分类号", "内容简介", "文献类型", "作品语种"]

RULE_TOKEN_MAP = {
    "r_vocational": "RULE_NEG_VOCATIONAL",
    "r_exam": "RULE_NEG_EXAM",
    "r_cartoon": "RULE_NEG_CARTOON",
    "r_gov_tourism": "RULE_NEG_GOV",
    "r_open_univ": "RULE_NEG_OPEN_UNIV",
    "r_micro_course": "RULE_NEG_MICROCOURSE",
    "r_engineering": "RULE_NEG_ENGINEERING",
    "r_audio_video_pub": "RULE_NEG_AUDIOVIDEO",
    "r_small_size": "RULE_NEG_SMALLSIZE",
    "r_few_pages": "RULE_NEG_FEWPAGES",
    "r_econ_soft": "RULE_NEG_ECON",
    "r_law_plain": "RULE_NEG_LAW",
    "r_shandong_qilu": "RULE_POS_QILU",
    "r_local_history": "RULE_POS_LOCALHISTORY",
    "r_expensive_history": "RULE_POS_EXPHISTORY",
    "r_marx_good": "RULE_POS_MARX",
    "r_folk_bigcity": "RULE_POS_FOLK",
    "r_edu_sport_quality": "RULE_POS_EDUSPORT",
    "r_green_architecture": "RULE_POS_GREENARCH",
}


def build_text_all(df: pd.DataFrame) -> pd.DataFrame:
    """根据字段 + 规则标签构造 text_all 列。"""
    def make_row_text(row):
        parts = []
        for col in TEXT_COLS:
            if col in row and pd.notna(row[col]):
                parts.append(str(row[col]))
        for col, token in RULE_TOKEN_MAP.items():
            if row.get(col, 0) == 1:
                parts.append(token)
        return " ".join(parts)

    df = df.copy()
    df["text_all"] = df.apply(make_row_text, axis=1)
    return df


def build_vectorizer() -> HashingVectorizer:
    """构造 HashingVectorizer，尝试优先用 jieba 分词。"""
    try:
        import jieba

        def zh_tokenizer(text):
            return [w.strip() for w in jieba.lcut(text) if w.strip()]

        tokenizer = zh_tokenizer
        token_pattern = None
        print("检测到 jieba，将使用中文分词。")
    except ImportError:
        tokenizer = None
        token_pattern = r"(?u)\b\w+\b"
        print("未检测到 jieba，使用默认分词。")

    vec = HashingVectorizer(
        n_features=2 ** 18,
        alternate_sign=False,
        tokenizer=tokenizer,
        token_pattern=token_pattern,
        norm="l2"
    )
    return vec


# ======================
# 模型初始化 / 训练 / 评估
# ======================

def init_or_load_model(model_path: str, class_weight: dict) -> tuple[SGDClassifier, bool]:
    """如果存在旧模型则加载，否则新建。返回 (模型, 是否新建)。"""
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            clf = pickle.load(f)
        print("已加载历史模型，将在此基础上继续训练。")
        return clf, False
    else:
        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            class_weight=class_weight
        )
        print("未发现历史模型，将创建新模型。")
        return clf, True


def train_model_incremental(
    X_all, y, model_path: str, class_weight: dict
):
    """
    使用增量学习在当前数据上训练模型：
    - 拆分 train / test
    - 对 train 做 partial_fit
    - 在 test 上评估
    - 再对 test 做 partial_fit
    - 保存模型
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, stratify=y, random_state=42
    )

    clf, is_new = init_or_load_model(model_path, class_weight)
    classes = np.array([0, 1])

    if is_new:
        clf.partial_fit(X_train, y_train, classes=classes)
    else:
        clf.partial_fit(X_train, y_train)

    # 评估
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "positive_rate": float(y.mean())
    }

    print("\n当前批次模型评估：")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 让模型完整吸收这批数据
    clf.partial_fit(X_test, y_test)

    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n模型已保存到：{model_path}")

    return clf, metrics


# ======================
# 打分、选书与评估命中率/查全率
# ======================

def score_and_recommend(df: pd.DataFrame, clf, vectorizer, threshold: float):
    """
    使用训练好的模型 + 规则，对全部书打分，并根据阈值选择推荐书目。
    返回 (带分数和推荐标记的 df, 推荐书目的子表, 推荐评估指标字典)
    """
    df = df.copy()

    # 模型分数
    X_all = vectorizer.transform(df["text_all"])
    df["score_model"] = clf.predict_proba(X_all)[:, 1]

    # 综合规则分数
    df["score_final"] = (
        df["score_model"]
        + CONFIG["pos_rule_weight"] * df["rule_pos_score"]
        - CONFIG["neg_rule_weight"] * df["rule_softneg_score"]
    )

    # 应用硬拒绝 + 阈值，得到“模型选中”的集合
    df["is_recommended"] = (
        (df["rule_hard_reject"] == 0) &
        (df["score_final"] >= threshold)
    ).astype(int)

    # 推荐子集
    rec_df = df[df["is_recommended"] == 1].copy()

    # 计算“命中率 / 查全率”
    y_true = df["is_selected"].values           # 实际是否订购
    y_pred = df["is_recommended"].values        # 模型是否推荐

    # 推荐中被实际选中的数量（TP）
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    rec_total = int((y_pred == 1).sum())
    true_total = int((y_true == 1).sum())

    hit_rate = tp / rec_total if rec_total > 0 else 0.0      # 命中率
    recall_rate = tp / true_total if true_total > 0 else 0.0 # 查全率

    rec_metrics = {
        "推荐总数": rec_total,
        "实际已订总数": true_total,
        "推荐且实际选中数量": tp,
        "命中率(precision)": hit_rate,
        "查全率(recall)": recall_rate
    }

    print("\n推荐效果评估（基于同一批数据模拟）：")
    for k, v in rec_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return df, rec_df, rec_metrics


# ======================
# 结果写 Excel
# ======================

def save_results(
    df_all: pd.DataFrame,
    df_rec: pd.DataFrame,
    model_metrics: dict,
    rec_metrics: dict,
    out_path: str
):
    """把全部打分结果、推荐书目、评估指标写入一个 Excel。"""
    metrics_df = pd.DataFrame(
        [{"指标": k, "数值": v} for k, v in model_metrics.items()]
    )
    rec_metrics_df = pd.DataFrame(
        [{"指标": k, "数值": v} for k, v in rec_metrics.items()]
    )

    # 统计每个规则被触发的次数
    rule_cols = [c for c in df_all.columns if c.startswith("r_")]
    rules_stat_df = pd.DataFrame({
        "规则名": rule_cols,
        "触发次数": df_all[rule_cols].sum().values
    })

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 全部书目（含标签与打分）
        cols_all = [
                       "ISBN",  # 新加这一列，便于后续对照
                       "题名", "责任者", "出版社", "分类号", "文献类型", "作品语种",
                       "价格", "出版年", "内容简介",
                       "is_selected", "score_model", "score_final","LLM_推荐分数","score_final_adj",
                       "rule_hard_reject", "rule_pos_score", "rule_softneg_score",
                       "is_recommended"
                   ] + rule_cols
        cols_all = [c for c in cols_all if c in df_all.columns]

        df_all[cols_all].to_excel(writer, sheet_name="全部书目_含打分与推荐", index=False)

        # 推荐书目
        rec_cols = [
            "ISBN",  # 新加这一列，便于后续对照
            "题名", "责任者", "出版社", "分类号", "文献类型", "作品语种",
            "价格", "出版年", "内容简介",
            "is_selected", "score_model", "score_final","LLM_推荐分数",
            "score_final_adj"
        ]
        rec_cols = [c for c in rec_cols if c in df_rec.columns]
        df_rec[rec_cols].to_excel(writer, sheet_name="推荐书目_模拟", index=False)

        # 模型评估
        metrics_df.to_excel(writer, sheet_name="模型评估(训练)", index=False)

        # 推荐命中率 / 查全率
        rec_metrics_df.to_excel(writer, sheet_name="推荐命中率_查全率", index=False)

        # 手工规则触发统计
        rules_stat_df.to_excel(writer, sheet_name="手工规则触发次数", index=False)

    print(f"\n结果已写入：{out_path}")



# 只读征订单的加载函数
def load_candidates(full_path: str,num:int=0) -> pd.DataFrame:
    """
    只读取征订单（征订记录合并表.xlsx），不加载订购明细、不打 is_selected 标签。
    用于：仅使用已有模型做选书推理的场景。
    """
    df = pd.read_excel(full_path,sheet_name=num)
    print("征订单待选书目记录数：", len(df))
    return df

# ======================
# 仅推理版：不训练，只用已有模型做选书模拟
# ======================
OUT_PATH_INFER = "模型选书结果_仅推理_征订单.xlsx"  # 仅推理模式输出文件名

def main_infer_only(FULL_PATH,num:int=0) -> pd.DataFrame:
    """
    不训练、不读取订购明细，只加载已经训练好的模型 book_selector_online.pkl，
    对当前征订单（征订记录合并表.xlsx）做一次选书，并输出推荐结果。
    注意：因为没有订购明细作为“真值”，本函数不计算命中率 / 查全率。
    """

    # 1. 只读取征订单，不加载订购明细
    df = load_candidates(FULL_PATH,num=num)

    # 2. 应用你的人工规则（高职不要、连环画不要、山东齐鲁优先等）
    df = apply_all_rules(df)

    # 3. 构造文本字段 text_all（题名+责任者+出版社+内容简介+规则 token）
    df = build_text_all(df)

    # 4. 构造向量器（HashingVectorizer 无状态，每次重建即可，需和训练时一致）
    vectorizer = build_vectorizer()

    # 5. 加载已经训练好的模型（不做任何训练）
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到已训练模型文件：{MODEL_PATH}\n"
            f"请先用 main() 训练一次，生成 book_selector_online.pkl。"
        )

    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f"已从 {MODEL_PATH} 加载训练好的模型，用于本次征订单选书。")

    # 6. 用模型 + 规则打分，并根据阈值选书
    #    这里我们不再依赖 is_selected，因此不用 score_and_recommend（它里头有命中率计算）
    df_scored = df.copy()

    X_all = vectorizer.transform(df_scored["text_all"])
    df_scored["score_model"] = clf.predict_proba(X_all)[:, 1]

    # 综合规则分数
    df_scored["score_final"] = (
        df_scored["score_model"]
        + CONFIG["pos_rule_weight"] * df["rule_pos_score"]
        - CONFIG["neg_rule_weight"] * df["rule_softneg_score"]
    )
    # # 应用硬规则 + 阈值，得到本次“模型推荐结果”
    # # score_final_adj = 0.7 * df_scored["score_final"] + 0.3 * df["LLM_推荐分数"]
    # df_scored["is_recommended"] = (
    #         (df_scored["rule_hard_reject"] == 0) &
    #         (df_scored["score_final"] >= RECOMMEND_THRESHOLD)
    # ).astype(int)


    # 机器学习与单模型综合打分
    df_scored["score_final_adj"] =CONFIG["machine_weight"] * df_scored["score_final"] + CONFIG["LLM_weight"] * df["LLM_推荐分数"]
    # 应用硬规则 + 阈值，得到本次“模型+机器推荐结果”
    df_scored["is_recommended"] = (
            (df_scored["rule_hard_reject"] == 0) &
            (df_scored["score_final_adj"] >= RECOMMEND_THRESHOLD)
    ).astype(int)


    # 推荐子集（就是你真正要看的候选书）
    df_rec = df_scored[df_scored["is_recommended"] == 1].copy()

    print("\n仅推理模式：")
    print("  征订单总数：", len(df_scored))
    print("  模型推荐数量：", len(df_rec))

    # 7. 写出结果到 Excel（这里只能输出推荐列表，不再有命中率/查全率）
    #    我们重用 save_results 的写法，不过模型评估只给一个说明，推荐评估为空。
    model_metrics = {
        "说明": "本次未训练、不使用订购明细，仅对征订单做模型推理选书。"
    }
    rec_metrics = {
        "说明": "本次推理未读取订购明细，因此无法计算命中率 / 查全率。"
    }

    # 可以复用 save_results，也可以简单写两张表；这里沿用原来的 save_results 方便你查看结构
    save_results(df_scored, df_rec, model_metrics, rec_metrics, OUT_PATH_INFER)

def evaluate_result_vs_order(
    result_path: str = "模型选书结果_模拟.xlsx",
    result_sheet: str = "全部书目_含打分与推荐",
    order_path: str = "订购明细-25LP07.xlsx",
    out_path: str = "模型选书_与订购明细对照评估.xlsx",
):
    """
    把“模型选书结果_模拟.xlsx”和“订购明细-25LP07.xlsx”进行对照，
    计算命中率（precision）和查全率（recall），并输出一份对照结果 Excel。

    计算规则（都以 ISBN 为单位去重）：
    - 推荐集合 R：模型推荐的 ISBN 集合（is_recommended == 1）
    - 实际集合 A：订购明细中的 ISBN 集合
    - TP = |R ∩ A|   推荐且实际订购
    - precision(命中率) = TP / |R|
    - recall(查全率)    = TP / |A|
    """

    # 1. 读取模型结果表
    res = pd.read_excel(result_path, sheet_name=result_sheet)

    if "ISBN" not in res.columns:
        raise ValueError(
            f"{result_path} 的工作表 {result_sheet} 中没有 'ISBN' 列。\n"
            f"请在生成模型结果时，把 ISBN 一并写入该表。"
        )
    if "is_recommended" not in res.columns:
        raise ValueError(
            f"{result_path} 的工作表 {result_sheet} 中没有 'is_recommended' 列，"
            f"无法区分哪些是模型推荐的书。"
        )

    # 2. 读取订购明细（注意 header=4）
    order = pd.read_excel(order_path, header=4)
    if "ISBN" not in order.columns:
        raise ValueError(f"{order_path} 中没有 'ISBN' 列，无法对照。")

    # 3. 归一化 ISBN
    res["ISBN_norm"] = res["ISBN"].apply(normalize_isbn)
    order["ISBN_norm"] = order["ISBN"].apply(normalize_isbn)

    # 4. 构建集合：模型推荐 & 实际订购
    rec_isbns = set(
        res.loc[res["is_recommended"] == 1, "ISBN_norm"].dropna().tolist()
    )
    actual_isbns = set(order["ISBN_norm"].dropna().tolist())

    # 只要 ISBN 非空，认为是本批次的候选全集
    all_isbns = set(res["ISBN_norm"].dropna().tolist())

    # 交集 / 差集
    tp_isbns = rec_isbns & actual_isbns            # 推荐且已订
    fp_isbns = rec_isbns - actual_isbns            # 推荐但未订
    fn_isbns = actual_isbns - rec_isbns            # 未推荐但已订
    tn_isbns = all_isbns - rec_isbns - actual_isbns  # 未推荐且未订（一般不太关心）

    TP = len(tp_isbns)
    REC_TOTAL = len(rec_isbns)
    TRUE_TOTAL = len(actual_isbns)

    precision = TP / REC_TOTAL if REC_TOTAL > 0 else 0.0   # 命中率
    recall = TP / TRUE_TOTAL if TRUE_TOTAL > 0 else 0.0    # 查全率

    print("\n=== 模型结果 vs 订购明细 对照评估 ===")
    print("模型推荐书目（去重ISBN）数量 |R|：", REC_TOTAL)
    print("实际订购书目（去重ISBN）数量 |A|：", TRUE_TOTAL)
    print("推荐且实际订购数量 |R∩A|：", TP)
    print(f"命中率 precision = {precision:.4f}")
    print(f"查全率 recall    = {recall:.4f}")

    # 5. 在模型结果表中标注“实际是否订购”以及对照情况
    res["actual_selected"] = res["ISBN_norm"].isin(actual_isbns).astype(int)

    def case_label(row):
        if row["is_recommended"] == 1 and row["actual_selected"] == 1:
            return "推荐且已订"
        elif row["is_recommended"] == 1 and row["actual_selected"] == 0:
            return "推荐未订"
        elif row["is_recommended"] == 0 and row["actual_selected"] == 1:
            return "未推荐但已订"
        else:
            return "未推荐且未订"

    res["对照情况"] = res.apply(case_label, axis=1)

    # 6. 指标汇总表
    metrics = {
        "模型推荐数量(|R|)": REC_TOTAL,
        "实际订购数量(|A|)": TRUE_TOTAL,
        "推荐且已订数量(|R∩A|)": TP,
        "命中率(precision)": precision,
        "查全率(recall)": recall,
        "FP_推荐未订数量": len(fp_isbns),
        "FN_未推荐但已订数量": len(fn_isbns),
        "TN_未推荐且未订数量": len(tn_isbns),
    }
    metrics_df = pd.DataFrame(
        [{"指标": k, "数值": v} for k, v in metrics.items()]
    )

    # 7. 输出结果 Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 对照明细（你可以筛选“推荐且已订 / 未推荐但已订”等）
        res.to_excel(writer, sheet_name="模型结果_与实际对照", index=False)

        # 指标汇总
        metrics_df.to_excel(writer, sheet_name="命中率_查全率_汇总", index=False)

    print(f"\n对照评估结果已写入：{out_path}")
    return metrics



# 原来的入口是训练 + 推理，如果你想在同一文件里切换，可以这样用：
if __name__ == "__main__":

    # FULL_PATH = "征订记录合并表.xlsx"  # 征订目录（全集）
    # FULL_PATH = "单模型征订记录合并表_智能推荐结果.xlsx"          # 单模型 单分类 LLM推荐结果表
    FULL_PATH = r"agents\outputs_1000_test\汇总智能推荐表.xlsx"
    SEL_PATH = "订购明细-25LP07.xlsx"                        # 已订明细 结果对照

    # 2. 只做推理（不更新模型）
    main_infer_only(FULL_PATH,num=1)


    # 3. 对照评估：用已有的“模型选书结果_模拟.xlsx”+“订购明细-25LP07.xlsx”
    evaluate_result_vs_order(
        result_path=OUT_PATH_INFER,    #模型推理后形成的表格
        result_sheet="全部书目_含打分与推荐",
        order_path=SEL_PATH,        #对照老师经验选取的表格
        out_path="单LLM选书_与订购明细对照评估.xlsx",       #生成对照结果的表格
    )

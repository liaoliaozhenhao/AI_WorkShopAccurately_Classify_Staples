import os
import re
import math
import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ======================
# 0. 基本路径配置
# ======================

FULL_PATH = "征订记录合并表.xlsx"          # 全部征订单（征订记录合并表）
SEL_PATH = "订购明细-25LP07.xlsx"        # 已订书目（订购明细）
MODEL_PATH = "book_selector_online.pkl"  # 自适应模型保存路径
OUT_PATH = "自适应选书模型结果.xlsx"      # 输出结果 Excel

CLASS_WEIGHT = {0: 1, 1: 10}             # 正负样本不平衡下，提升“选中”这一类的权重


# ======================
# 1. 读数据 & 打标签
# ======================

def normalize_isbn(x):
    if pd.isna(x):
        return ""
    return str(x).replace("-", "").strip()


full = pd.read_excel(FULL_PATH)
sel = pd.read_excel(SEL_PATH, header=4)  # 你的订购明细前面有说明行

full["ISBN_norm"] = full["ISBN"].apply(normalize_isbn)
sel["ISBN_norm"] = sel["ISBN"].apply(normalize_isbn)

sel_isbns = set(sel["ISBN_norm"])
full["is_selected"] = full["ISBN_norm"].isin(sel_isbns).astype(int)

print("征订单总记录：", len(full))
print("已订中记录数：", full["is_selected"].sum())


# ======================
# 2. 一些通用小工具函数
# ======================

def concat_text(row, cols):
    """把若干列合并成一个字符串，用于规则匹配。"""
    parts = []
    for c in cols:
        if c in row and pd.notna(row[c]):
            parts.append(str(row[c]))
    return " ".join(parts)


def contains_any(text, keywords):
    if not isinstance(text, str):
        return False
    return any(k in text for k in keywords)


def parse_size_cm(val):
    """从 '21cm' 之类的字符串里提取数值。"""
    if pd.isna(val):
        return np.nan
    s = str(val)
    m = re.search(r"(\d+(\.\d+)?)\s*cm", s)
    if m:
        return float(m.group(1))
    return np.nan


# ======================
# 3. 把订书原则写成规则函数
# ======================

# 3.1 明确不要的（硬拒绝）

def rule_vocational(row):
    text = concat_text(row, ["题名", "内容简介", "备注"])
    return contains_any(text, ["高职", "高专", "职业学院", "职业教育", "中职", "技工", "中等职业"])


def rule_small_size(row):
    # 只有有“尺寸 / 开本”列才会生效
    size_col = None
    if "尺寸" in row:
        size_col = row["尺寸"]
    elif "开本" in row:
        size_col = row["开本"]
    size_cm = parse_size_cm(size_col) if size_col is not None else np.nan
    return (not np.isnan(size_cm)) and (size_cm < 19)


def rule_few_pages(row, threshold=120):
    # 只有有“页码 / 页数”列才会生效
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
    text = concat_text(row, ["题名", "内容简介", "备注"])
    kws = ["连环画", "漫画", "绘本", "图画书", "宣传画", "卡通"]
    return contains_any(text, kws)


def rule_exam(row):
    text = concat_text(row, ["题名", "内容简介", "备注"])
    kws = [
        "考试", "考点", "真题", "题库", "押题", "冲刺", "模拟试题", "试题解析",
        "历年试题", "辅导", "精讲精练", "过关", "考研", "国考", "省考",
        "教师资格", "资格考试", "职业资格", "注册"
    ]
    return contains_any(text, kws)


def rule_gov_tourism(row):
    text = concat_text(row, ["题名", "内容简介", "出版社"])
    kws = [
        "旅游厅", "文化和旅游厅", "人民政府", "市政府", "县政府", "自治区人民政府",
        "州人民政府", "政务", "政府公报", "政协", "人大常委会"
    ]
    return contains_any(text, kws)


def rule_open_univ(row):
    pub = str(row.get("出版社", "") or "")
    return "国家开放大学出版社" in pub


def rule_micro_course(row):
    text = concat_text(row, ["题名", "内容简介"])
    return ("微课" in text) or ("慕课" in text)


def rule_engineering(row):
    text = concat_text(row, ["题名", "内容简介", "分类号"])
    kws = ["施工", "土木工程", "水利工程", "桥梁工程", "隧道工程", "公路工程", "岩土工程", "市政工程"]
    return contains_any(text, kws)


def rule_audio_video_pub(row):
    pub = str(row.get("出版社", "") or "")
    return ("电子音像" in pub) or ("音像出版社" in pub)


# 3.2 软惩罚 & 软加分规则

def rule_shandong_qilu(row):
    text = concat_text(row, ["题名", "内容简介", "出版社"])
    kws = ["山东", "齐鲁", "齐鲁文化"]
    return contains_any(text, kws)


def rule_local_history(row):
    text = concat_text(row, ["题名", "内容简介"])
    kws = ["地方志", "市志", "县志", "区志", "乡镇志", "乡志", "村志", "街道志", "县情", "市情"]
    return contains_any(text, kws)


def rule_expensive_history(row, price_threshold=200):
    try:
        price = float(row.get("价格", np.nan))
    except Exception:
        price = np.nan
    class_no = str(row.get("分类号", "") or "")
    is_history = class_no.startswith("K")
    return is_history and (not np.isnan(price)) and price > price_threshold


def rule_econ_soft(row):
    class_no = str(row.get("分类号", "") or "")
    return class_no.startswith("F")  # 经济类尽量不要


def rule_law_plain(row):
    class_no = str(row.get("分类号", "") or "")
    is_law = class_no.startswith("D9") or class_no.startswith("D92") or class_no.startswith("D93")
    text = concat_text(row, ["题名", "内容简介"])
    good_kws = ["法律史", "法制史", "法学理论", "法律理论", "法哲学"]
    if is_law and not contains_any(text, good_kws):
        return True  # 一般法律类，少订
    return False


def rule_marx_good(row):
    text = concat_text(row, ["题名", "内容简介"])
    pub = str(row.get("出版社", "") or "")
    has_marx = ("马克思" in text) or ("马列" in text) or ("马克思主义" in text)
    good_pubs = ["人民出版社", "高等教育出版社", "中国人民大学出版社", "商务印书馆", "中国社会科学出版社"]
    return has_marx and any(gp in pub for gp in good_pubs)


def rule_folk_bigcity(row):
    text = concat_text(row, ["题名", "内容简介"])
    if "民俗" not in text:
        return False
    big_cities = ["上海", "北京", "广州", "深圳", "南京", "杭州", "成都", "重庆", "天津", "武汉", "西安"]
    return contains_any(text, big_cities)


def rule_edu_sport_quality(row):
    text = concat_text(row, ["题名", "内容简介"])
    has_edu_sport = ("教育" in text) or ("体育" in text)
    pub = str(row.get("出版社", "") or "")
    good_pubs = ["高等教育出版社", "人民教育出版社", "北京体育大学出版社",
                 "上海体育学院出版社", "北京师范大学出版社"]
    return has_edu_sport and any(gp in pub for gp in good_pubs)


def rule_green_architecture(row):
    text = concat_text(row, ["题名", "内容简介"])
    has_arch = "建筑" in text
    kws = ["绿色建筑", "绿色", "可持续", "低碳", "生态", "节能", "环境友好"]
    return has_arch and contains_any(text, kws)


# ======================
# 4. 在 DataFrame 上应用规则
# ======================

full["r_vocational"] = full.apply(rule_vocational, axis=1).astype(int)
full["r_small_size"] = full.apply(rule_small_size, axis=1).astype(int)
full["r_few_pages"] = full.apply(rule_few_pages, axis=1).astype(int)
full["r_cartoon"] = full.apply(rule_cartoon, axis=1).astype(int)
full["r_exam"] = full.apply(rule_exam, axis=1).astype(int)
full["r_gov_tourism"] = full.apply(rule_gov_tourism, axis=1).astype(int)
full["r_open_univ"] = full.apply(rule_open_univ, axis=1).astype(int)
full["r_micro_course"] = full.apply(rule_micro_course, axis=1).astype(int)
full["r_engineering"] = full.apply(rule_engineering, axis=1).astype(int)
full["r_audio_video_pub"] = full.apply(rule_audio_video_pub, axis=1).astype(int)

full["r_shandong_qilu"] = full.apply(rule_shandong_qilu, axis=1).astype(int)
full["r_local_history"] = full.apply(rule_local_history, axis=1).astype(int)
full["r_expensive_history"] = full.apply(rule_expensive_history, axis=1).astype(int)
full["r_econ_soft"] = full.apply(rule_econ_soft, axis=1).astype(int)
full["r_law_plain"] = full.apply(rule_law_plain, axis=1).astype(int)
full["r_marx_good"] = full.apply(rule_marx_good, axis=1).astype(int)
full["r_folk_bigcity"] = full.apply(rule_folk_bigcity, axis=1).astype(int)
full["r_edu_sport_quality"] = full.apply(rule_edu_sport_quality, axis=1).astype(int)
full["r_green_architecture"] = full.apply(rule_green_architecture, axis=1).astype(int)

# 硬拒绝：有一个 True 就拒绝
hard_cols = [
    "r_vocational", "r_small_size", "r_few_pages", "r_cartoon",
    "r_gov_tourism", "r_open_univ", "r_micro_course",
    "r_engineering", "r_exam"
]
full["rule_hard_reject"] = (full[hard_cols].sum(axis=1) > 0).astype(int)

# 软加分 / 软减分
pos_cols = [
    "r_shandong_qilu", "r_local_history", "r_marx_good",
    "r_folk_bigcity", "r_edu_sport_quality", "r_green_architecture"
]
neg_soft_cols = ["r_econ_soft", "r_law_plain", "r_audio_video_pub"]

full["rule_pos_score"] = full[pos_cols].sum(axis=1)
full["rule_softneg_score"] = full[neg_soft_cols].sum(axis=1)


# ======================
# 5. 构造文本特征（含规则 token）
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


def make_text_all(row):
    parts = []
    for col in TEXT_COLS:
        if col in row and pd.notna(row[col]):
            parts.append(str(row[col]))
    # 加上规则标签
    for col, token in RULE_TOKEN_MAP.items():
        if row.get(col, 0) == 1:
            parts.append(token)
    return " ".join(parts)


full["text_all"] = full.apply(make_text_all, axis=1)


# ======================
# 6. 文本向量化 & 自适应模型（增量学习）
# ======================

# 若环境中有 jieba，可以自动使用中文分词；否则用默认分词方案
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
    print("未检测到 jieba，使用默认分词（也能用）。")

vectorizer = HashingVectorizer(
    n_features=2 ** 18,     # 262144 维，足够大
    alternate_sign=False,   # 方便解释 & 稳定
    tokenizer=tokenizer,
    token_pattern=token_pattern,
    norm="l2"
)

X_all = vectorizer.transform(full["text_all"])
y = full["is_selected"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.array([0, 1])

# 载入旧模型（如果有），否则创建新模型
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    is_new_model = False
    print("已加载历史模型，将在此基础上继续训练。")
else:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        class_weight=CLASS_WEIGHT
    )
    is_new_model = True
    print("未发现历史模型，将创建新模型。")

# 增量训练
if is_new_model:
    clf.partial_fit(X_train, y_train, classes=classes)
else:
    clf.partial_fit(X_train, y_train)

# 在当前批次上做一个简单评估（仅用来看看效果）
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
    "positive_rate": y.mean()
}

print("\n当前批次模型评估：")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 再用测试集也做一次增量训练，让模型完整吸收当前批次数据
clf.partial_fit(X_test, y_test)

# 保存模型，供下次在此基础上继续训练
with open(MODEL_PATH, "wb") as f:
    pickle.dump(clf, f)
print(f"\n模型已保存到：{MODEL_PATH}")

# 用更新后的模型给所有书打分
full["score_model"] = clf.predict_proba(X_all)[:, 1]


# ======================
# 7. 综合规则分数 & 最终推荐
# ======================

# 综合分数：模型分数 + 规则软加减分
full["score_final"] = (
    full["score_model"]
    + 0.2 * full["rule_pos_score"]
    - 0.1 * full["rule_softneg_score"]
)

# 只在：未选中过、且不违反硬规则的书里做推荐
candidates = full[(full["is_selected"] == 0) & (full["rule_hard_reject"] == 0)].copy()
candidates = candidates.sort_values("score_final", ascending=False)


def suggest_copies(row):
    """根据你的原则给出一个简单的“建议订购册数”方案。"""
    # 山东 / 齐鲁文化：建议 3 本
    if row["r_shandong_qilu"] == 1:
        return 3
    # 地方志 / 县乡村志：建议 1 本
    if row["r_local_history"] == 1:
        return 1
    # 贵价史料：只订一本
    if row["r_expensive_history"] == 1:
        return 1
    # 其他：可以按分数调节数量，这里简单示例：分数特别高订两本
    if row["score_final"] >= 0.8:
        return 2
    return 1


candidates["建议订购册数"] = candidates.apply(suggest_copies, axis=1).astype(int)

recommend_top200 = candidates.head(200)  # 例如取前 200 本作为推荐列表


# ======================
# 8. 写出 Excel 结果
# ======================

metrics_df = pd.DataFrame(
    [{"指标": k, "数值": v} for k, v in metrics.items()]
)

rule_cols = list(RULE_TOKEN_MAP.keys())
rules_stat_df = pd.DataFrame({
    "规则名": rule_cols,
    "触发次数": full[rule_cols].sum().values
})

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    # 全部书目 + 规则 + 模型分数
    cols_all = [
        "题名", "责任者", "出版社", "分类号", "文献类型", "作品语种",
        "价格", "出版年", "内容简介",
        "is_selected", "score_model", "score_final",
        "rule_hard_reject", "rule_pos_score", "rule_softneg_score"
    ] + rule_cols
    cols_all = [c for c in cols_all if c in full.columns]
    full[cols_all].to_excel(writer, sheet_name="全部书目_含规则与打分", index=False)

    # 推荐书目
    rec_cols = [
        "题名", "责任者", "出版社", "分类号", "文献类型", "作品语种",
        "价格", "出版年", "内容简介",
        "score_model", "score_final", "建议订购册数"
    ]
    rec_cols = [c for c in rec_cols if c in recommend_top200.columns]
    recommend_top200[rec_cols].to_excel(writer, sheet_name="推荐书目Top200", index=False)

    # 模型评估
    metrics_df.to_excel(writer, sheet_name="模型评估(本批)", index=False)

    # 手工规则触发统计
    rules_stat_df.to_excel(writer, sheet_name="手工规则触发次数", index=False)

print(f"\n分析完成，结果已写入：{OUT_PATH}")

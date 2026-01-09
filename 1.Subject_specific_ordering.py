import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ======================
# 0. 基本配置
# ======================

FULL_PATH = "征订记录合并表.xlsx"        # 全部征订单
SEL_PATH = "订购明细-25LP07.xlsx"      # 已订明细
OUT_PATH = "智能选书模型分析.xlsx"      # 输出结果


# ======================
# 1. 读取 & 对齐两个表
# ======================

def normalize_isbn(x):
    """把 ISBN 统一成不带横线的字符串."""
    if pd.isna(x):
        return ""
    return str(x).replace("-", "").strip()


# 1.1 读取征订目录（全集）
full = pd.read_excel(FULL_PATH)
full["ISBN_norm"] = full["ISBN"].apply(normalize_isbn)

# 1.2 读取已订明细（注意 header=4）
sel = pd.read_excel(SEL_PATH, header=4)
sel["ISBN_norm"] = sel["ISBN"].apply(normalize_isbn)

sel_isbns = set(sel["ISBN_norm"])

full["is_selected"] = full["ISBN_norm"].isin(sel_isbns).astype(int)

print("征订单总记录：", len(full))
print("已订中记录数：", full["is_selected"].sum())


# ======================
# 2. 处理出版年 & 价格分段标签
# ======================

def extract_year(x):
    """从各种格式中提取年份（前4位数字）."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    for ch in ["年", "月", "日", ".", "-", "/", " "]:
        s = s.replace(ch, "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 4:
        return int(digits[:4])
    return np.nan

full["出版年_数值"] = full["出版年"].apply(extract_year)


def price_tag(price):
    """把价格变成几个粗略区间标签."""
    if pd.isna(price):
        return ""
    try:
        p = float(price)
    except ValueError:
        return ""
    if p < 50:
        return "价格_0_50"
    elif p < 80:
        return "价格_50_80"
    elif p < 120:
        return "价格_80_120"
    else:
        return "价格_120_plus"


def year_tag(year):
    """把出版年变成几个粗略区间标签."""
    if pd.isna(year):
        return ""
    y = int(year)
    if y >= 2023:
        return "出版年_2023plus"
    elif y >= 2020:
        return "出版年_2020_2022"
    elif y >= 2015:
        return "出版年_2015_2019"
    else:
        return "出版年_2014_before"


# ======================
# 3. 构造“综合文本字段” text_all
# ======================

TEXT_COLS = ["题名", "责任者", "出版社", "作品语种", "文献类型", "内容简介"]

def make_text(row):
    parts = []
    for col in TEXT_COLS:
        val = row.get(col, "")
        if pd.isna(val):
            continue
        parts.append(str(val))
    # 加入价格和出版年的标签（不是原始数值，而是“词”）
    parts.append(price_tag(row.get("价格", np.nan)))
    parts.append(year_tag(row.get("出版年_数值", np.nan)))
    # 最终合并成一段文本
    return " ".join([p for p in parts if p])

full["text_all"] = full.apply(make_text, axis=1)


# ======================
# 4. 构建 TF-IDF + 逻辑回归 模型
# ======================

# 尝试使用 jieba 做中文分词（可选）
try:
    import jieba

    def chinese_tokenizer(text):
        return [w.strip() for w in jieba.lcut(text) if w.strip()]

    tokenizer = chinese_tokenizer
    token_pattern = None   # 自定义 tokenizer 时要把 token_pattern 设为 None
    print("已检测到 jieba，将使用中文分词。")

except ImportError:
    tokenizer = None
    token_pattern = r"(?u)\b\w+\b"
    print("未检测到 jieba，将使用默认分词（可能略差但能跑）。")


vectorizer = TfidfVectorizer(
    max_features=30000,          # 可视实际数据量调小或调大
    ngram_range=(1, 2),          # 1-gram + 2-gram，能抓到短语
    tokenizer=tokenizer,
    token_pattern=token_pattern
)

# 特征矩阵 X、标签 y
X_text = vectorizer.fit_transform(full["text_all"])
y = full["is_selected"].values

# ======================
# 5. 划分训练 / 测试集，评估模型
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"  # 正负样本比例不均时有用
)

clf.fit(X_train, y_train)

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

print("\n模型评估指标：")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# 用全部数据重新训练一个最终模型（用于打分）
clf.fit(X_text, y)
full["score_model"] = clf.predict_proba(X_text)[:, 1]


# ======================
# 6. 从模型中抽取“关键词规则”
# ======================

feature_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_[0]

# 权重从小到大排序
sorted_idx = np.argsort(coefs)

# 最“减分”的 50 个特征
neg_idx = sorted_idx[:50]
neg_words = feature_names[neg_idx]
neg_weights = coefs[neg_idx]

# 最“加分”的 50 个特征
pos_idx = sorted_idx[-50:]
pos_words = feature_names[pos_idx]
pos_weights = coefs[pos_idx]

# 生成 DataFrame 方便写入 Excel
pos_df = pd.DataFrame({
    "关键词": pos_words[::-1],
    "权重": pos_weights[::-1]
})
pos_df["解释"] = "权重越大，包含该词的图书越容易被模型判为“应选”"

neg_df = pd.DataFrame({
    "关键词": neg_words,
    "权重": neg_weights
})
neg_df["解释"] = "权重越小（越负），包含该词的图书越不容易被模型判为“应选”"


# ======================
# 7. 生成“推荐书目”列表（未选部分按模型分数排序）
# ======================

# 已选的是你已经订过的，未选可作为下一轮候选
candidates = full[full["is_selected"] == 0].copy()
candidates = candidates.sort_values("score_model", ascending=False)

# 例如取前 200 本作为推荐示例
recommend_top200 = candidates.head(200)


# ======================
# 8. 整理模型评估指标为表格
# ======================

metrics_df = pd.DataFrame(
    [{"指标": k, "数值": v} for k, v in metrics.items()]
)


# ======================
# 9. 写出到 Excel
# ======================

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    # 全部书目 + 标签 + 模型分数
    full[[
        "题名", "责任者", "出版社", "作品语种", "文献类型",
        "价格", "出版年", "出版年_数值",
        "内容简介",
        "is_selected", "score_model"
    ]].to_excel(writer, sheet_name="全部书目_含模型分数", index=False)

    # 推荐书目
    recommend_top200[[
        "题名", "责任者", "出版社", "作品语种", "文献类型",
        "价格", "出版年", "内容简介", "score_model"
    ]].to_excel(writer, sheet_name="推荐书目Top200", index=False)

    # 关键词权重（正向/负向）
    pos_df.to_excel(writer, sheet_name="偏好关键词_正向", index=False)
    neg_df.to_excel(writer, sheet_name="偏好关键词_负向", index=False)

    # 模型评估
    metrics_df.to_excel(writer, sheet_name="模型评估", index=False)

print(f"\n分析完成，结果已写入：{OUT_PATH}")

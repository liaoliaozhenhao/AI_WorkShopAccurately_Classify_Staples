# -*- coding: utf-8 -*-
"""
多模型对比：用《征订记录合并表.xlsx》训练“已订/未订”二分类模型

你要的效果（对齐 Q1.py 的思路）：
1) 读取 Excel
2) “状态(已订/未订)”当作标签 y
3) 其余列当作特征 X（自动把列拆成：文本列/类别列/数值列）
4) 训练多个模型（线性/非线性）并对比性能
5) 输出：对比表、最佳模型、预测结果 Excel、ROC/PR/混淆矩阵图

依赖：
pip install pandas numpy scikit-learn openpyxl joblib matplotlib

运行示例：
python order_status_multimodel.py --input "征订记录合并表.xlsx" --label_col "状态" --output_dir "outputs_q1_style"

可选：
--mode tfidf   # tfidf 或 hashing，默认 hashing 更快更省内存
--select_metric f1_pos  # 用哪个指标选“最佳模型”
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------
# 1) 数据读入与列类型拆分
# ---------------------------
def load_data(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def infer_label_column(df: pd.DataFrame, preferred: Optional[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    # 兜底：找只包含“已订/未订”的列
    for col in df.columns:
        vals = df[col].dropna().astype(str).unique()
        if len(vals) == 2 and set(vals) == {"已订", "未订"}:
            return col
    raise ValueError("未找到标签列。请用 --label_col 显式指定，例如：--label_col 状态")


def _combine_text_matrix(X) -> np.ndarray:
    """
    把多列文本拼成一个字符串列，交给向量化器。
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    out = []
    for row in X:
        parts = []
        for v in row:
            if v is None:
                continue
            sv = str(v)
            if sv == "nan":
                continue
            parts.append(sv)
        out.append(" ".join(parts))
    return np.array(out, dtype=object)


def split_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], List[str]]:
    """
    自动把特征列拆成：
    - text_cols: 长文本或高基数列（题名/内容简介/备注/ISBN/责任者...）
    - cat_cols : 低基数类别列（文献类型/语种/载体/币种...）
    - num_cols : 数值列（价格/出版年/预订套数...）
    """
    df = df.copy()
    drop_cols = drop_cols or []
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # 删除全空列
    all_null_cols = [c for c in df.columns if df[c].isna().all()]
    if all_null_cols:
        df = df.drop(columns=all_null_cols)

    if label_col not in df.columns:
        raise ValueError(f"标签列 {label_col} 不存在")

    y_raw = df[label_col].astype(str)
    y = y_raw.map({"已订": 1, "未订": 0}).to_numpy()
    if np.isnan(y).any():
        bad = sorted(set(y_raw[~y_raw.isin(["已订", "未订"])].unique().tolist()))
        raise ValueError(f"标签列 {label_col} 存在非“已订/未订”的值：{bad}")

    X = df.drop(columns=[label_col])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]

    text_cols, cat_cols = [], []
    for c in obj_cols:
        s = X[c].dropna().astype(str).head(3000)
        avg_len = float(s.map(len).mean()) if len(s) else 0.0
        nunique = int(s.nunique())

        # 启发式：长文本 或 高基数 → text
        if avg_len >= 12 or nunique > 60:
            text_cols.append(c)
        else:
            cat_cols.append(c)

    return X, y, text_cols, cat_cols, num_cols


# ---------------------------
# 2) 预处理器：统一特征抽取
# ---------------------------
def build_preprocess(
    text_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    mode: str = "hashing",
) -> ColumnTransformer:
    """
    mode:
      - hashing: HashingVectorizer(char ngram) 速度快、内存友好（但不便解释）
      - tfidf  : TfidfVectorizer(char ngram) 可解释性更好（可能更慢/更占内存）
    """
    if mode == "hashing":
        vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
        )
    elif mode == "tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=60000,
        )
    else:
        raise ValueError("mode 只能是 hashing 或 tfidf")

    text_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("combine", FunctionTransformer(_combine_text_matrix, validate=False)),
            ("vec", vectorizer),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_cols),
            ("cat", cat_transformer, cat_cols),
            ("num", num_transformer, num_cols),
        ],
        remainder="drop",
    )
    return preprocess


# ---------------------------
# 3) 评估：ROC/PR/混淆矩阵
# ---------------------------
def get_score_vector(model, X) -> np.ndarray:
    """
    统一得到“正类=已订”的连续分数，用于 ROC-AUC/PR-AUC/画曲线：
    - 有 predict_proba -> 用 proba[:,1]
    - 有 decision_function -> 用 decision_function（再通过 min-max 缩放到0-1便于阈值）
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s).ravel()
        # min-max 缩放到 0-1（仅用于绘图/阈值统一）
        mn, mx = float(s.min()), float(s.max())
        if mx - mn < 1e-12:
            return np.zeros_like(s, dtype=float)
        return (s - mn) / (mx - mn)
    raise ValueError("模型既没有 predict_proba，也没有 decision_function")


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def plot_roc_pr_cm(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray, out_dir: Path, name: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"roc_{name}.png", dpi=150)
    plt.close()

    # PR
    p, r, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {name}")
    plt.tight_layout()
    plt.savefig(out_dir / f"pr_{name}.png", dpi=150)
    plt.close()

    # Confusion Matrix（简单绘图）
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {name}")
    plt.xticks([0, 1], ["未订", "已订"])
    plt.yticks([0, 1], ["未订", "已订"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / f"cm_{name}.png", dpi=150)
    plt.close()


# ---------------------------
# 4) 多模型训练与对比（Q1.py 风格）
# ---------------------------
def build_models(preprocess: ColumnTransformer, random_state: int = 42) -> Dict[str, Pipeline]:
    """
    设计两类模型：
    A. 直接吃稀疏特征（文本TFIDF/Hashing + onehot）→ 线性模型更稳
    B. 稀疏特征先 SVD 降维到低维稠密 → 可以上 RF / GBDT / MLP
    """
    # A类：稀疏输入
    lr = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(
            solver="saga",
            max_iter=2000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    linsvc = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LinearSVC(
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    sgd = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", SGDClassifier(
            loss="log_loss",
            max_iter=3000,
            tol=1e-3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    # B类：SVD降维后再模型
    svd_block = Pipeline(steps=[
        ("preprocess", preprocess),
        ("svd", TruncatedSVD(n_components=200, random_state=random_state)),
        ("scaler", StandardScaler()),
    ])

    rf = Pipeline(steps=[
        ("features", svd_block),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])

    gbdt = Pipeline(steps=[
        ("features", svd_block),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        )),
    ])

    mlp = Pipeline(steps=[
        ("features", svd_block),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=50,
            random_state=random_state,
            early_stopping=True,
        )),
    ])

    return {
        "LogisticRegression": lr,
        "LinearSVC": linsvc,
        "SGD(log_loss)": sgd,
        "RandomForest(SVD)": rf,
        "GBDT(SVD)": gbdt,
        "MLP(SVD)": mlp,
    }


def train_compare_models(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    out_dir: Path,
    select_metric: str = "f1_pos",
    test_size: float = 0.25,
    random_state: int = 42,
    do_cv: bool = False,
    cv_splits: int = 5,
    cv_scoring: str = "accuracy",
) -> Tuple[str, Pipeline, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    rows = []
    best_name, best_model = None, None
    best_score = -1e9

    for name, model in models.items():
        print(f"\n===== 训练 {name} =====")
        model.fit(X_train, y_train)

        # 预测
        y_score = get_score_vector(model, X_test)
        y_pred = (y_score >= 0.5).astype(int)  # 统一使用 0.5 阈值（可后续再做阈值优化）

        metrics = evaluate_binary(y_test, y_pred, y_score)
        print(classification_report(y_test, y_pred, target_names=["未订", "已订"], digits=4))
        print(f"[{name}] metrics:", {k: metrics[k] for k in ["accuracy","precision_pos","recall_pos","f1_pos","roc_auc","pr_auc"]})

        # 曲线/混淆矩阵图
        safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        plot_roc_pr_cm(y_test, y_score, y_pred, plots_dir, safe_name)

        # 交叉验证（默认用 accuracy；你也可以改 scoring）
        cv_mean, cv_std = np.nan, np.nan
        if do_cv:
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1)
                cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
                print(f"[{name}] {cv_splits}折CV({cv_scoring}): {cv_mean:.4f} ± {cv_std:.4f}")
            except Exception as e:
                print(f"[WARN] {name} 交叉验证失败：{e}")

        row = {"model": name, **metrics, "cv_mean": cv_mean, "cv_std": cv_std}
        rows.append(row)

        sel = float(metrics.get(select_metric, -1e9))
        if sel > best_score:
            best_score = sel
            best_name, best_model = name, model

    results_df = pd.DataFrame(rows).sort_values(by=select_metric, ascending=False)
    results_df.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # 画一个对比柱状图（选定指标）
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["model"].astype(str), results_df[select_metric].astype(float))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(select_metric)
    plt.title(f"Model comparison by {select_metric}")
    plt.tight_layout()
    plt.savefig(out_dir / f"compare_{select_metric}.png", dpi=150)
    plt.close()

    print(f"\n[BEST] 按 {select_metric} 最佳模型：{best_name} = {best_score:.4f}")
    return best_name, best_model, results_df


def export_best_model_predictions(
    df_raw: pd.DataFrame,
    label_col: str,
    best_name: str,
    best_model: Pipeline,
    drop_cols: List[str],
    out_dir: Path,
):
    X_all = df_raw.drop(columns=[label_col], errors="ignore")
    X_all = X_all.drop(columns=[c for c in drop_cols if c in X_all.columns], errors="ignore")

    y_score = get_score_vector(best_model, X_all)
    y_pred = (y_score >= 0.5).astype(int)

    df_out = df_raw.copy()
    df_out["pred_prob_已订"] = y_score
    df_out["pred_状态"] = np.where(y_pred == 1, "已订", "未订")

    # 如果原始数据有标签，就输出是否命中
    if label_col in df_out.columns:
        y_true = df_out[label_col].astype(str).map({"已订": 1, "未订": 0}).to_numpy()
        if not np.isnan(y_true).any():
            df_out["pred_correct"] = (y_true == y_pred)

    out_path = out_dir / "征订记录_最佳模型预测结果.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"[OK] 已导出预测结果：{out_path}")

    # 保存模型
    model_path = out_dir / "best_model.joblib"
    joblib.dump({"model_name": best_name, "pipeline": best_model, "label_col": label_col, "drop_cols": drop_cols}, model_path)
    print(f"[OK] 已保存最佳模型：{model_path}")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", required=True, help="输入Excel，如：征订记录合并表.xlsx")
    # parser.add_argument("--label_col", default="状态", help="标签列名，默认：状态")
    # parser.add_argument("--mode", default="hashing", choices=["hashing", "tfidf"], help="文本向量化方式")
    # parser.add_argument("--output_dir", default="outputs_q1_style", help="输出目录")
    # parser.add_argument("--select_metric", default="f1_pos",
    #                     choices=["accuracy", "precision_pos", "recall_pos", "f1_pos", "roc_auc", "pr_auc"],
    #                     help="用于选最佳模型的指标")
    # parser.add_argument("--do_cv", type=int, default=0,
    #                     help="是否对每个模型做交叉验证（1=是，0=否）。注意：会显著变慢。")
    # parser.add_argument("--cv_splits", type=int, default=5, help="交叉验证折数")
    # parser.add_argument("--cv_scoring", type=str, default="accuracy",
    #                     help="交叉验证评分：accuracy / f1 / f1_macro / roc_auc 等（与sklearn一致）")
    # parser.add_argument("--test_size", type=float, default=0.25)
    # parser.add_argument("--random_state", type=int, default=42)
    # parser.add_argument("--drop_cols", default="序号", help="要丢弃的列（逗号分隔），默认丢弃“序号”")
    # args = parser.parse_args()
    #
    # out_dir = Path(args.output_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)
    #
    # df = load_data(args.input)
    # label_col = infer_label_column(df, args.label_col)
    #
    # drop_cols = [c.strip() for c in str(args.drop_cols).split(",") if c.strip()]
    # X, y, text_cols, cat_cols, num_cols = split_feature_columns(df, label_col, drop_cols=drop_cols)
    #
    # print(f"[INFO] 样本数: {len(df)}")
    # print(f"[INFO] 标签列: {label_col}（已订=1, 未订=0）")
    # print(f"[INFO] 正例(已订)占比: {y.mean():.4%}")
    # print(f"[INFO] text_cols({len(text_cols)}): {text_cols}")
    # print(f"[INFO] cat_cols({len(cat_cols)}): {cat_cols}")
    # print(f"[INFO] num_cols({len(num_cols)}): {num_cols}")

    # ==================== 在这里修改你的配置 ====================
    INPUT_FILE = "征订记录合并表.xlsx"  # 修改为你的Excel文件路径
    LABEL_COL = "状态"  # 标签列名
    MODE = "hashing"  # 向量化方式: hashing 或 tfidf
    OUTPUT_DIR = "outputs_q1_styLe"  # 输出目录
    SELECT_METRIC = "f1_pos"  # 模型选择指标
    DO_CV = 0  # 是否交叉验证: 1=是, 0=否
    CV_SPLITS = 5  # 交叉验证折数
    CV_SCORING = "accuracy"  # 交叉验证评分
    TEST_SIZE = 0.25  # 测试集比例
    RANDOM_STATE = 42  # 随机种子
    DROP_COLS = "序号"  # 要丢弃的列（逗号分隔）
    # =========================================================

    # 创建输出目录
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df = load_data(INPUT_FILE)

    # 推断标签列（修正了原代码的拼写错误）
    label_col = infer_label_column(df, LABEL_COL)

    # 处理要删除的列
    drop_cols = [c.strip() for c in str(DROP_COLS).split(",") if c.strip()]

    # 分割特征列
    X, y, text_cols, cat_cols, num_cols = split_feature_columns(df, label_col, drop_cols=drop_cols)

    # 打印信息（修正了原代码的拼写错误）
    print(f"[INFO] 样本数：{len(df)}")
    print(f"[INFO] 标签列：{label_col}（己订=1，未订=0）")
    print(f"[INFO] 正例(已订)占比：{y.mean():.4%}")
    print(f"[INFO] text_cols({len(text_cols)}): {text_cols}")
    print(f"[INFO] cat_cols({len(cat_cols)}): {cat_cols}")
    print(f"[INFO] num_cols({len(num_cols)}): {num_cols}")


    preprocess = build_preprocess(text_cols, cat_cols, num_cols, mode=MODE)
    models = build_models(preprocess, random_state=RANDOM_STATE)

    best_name, best_model, results_df = train_compare_models(
        X=X, y=y, models=models, out_dir=out_dir,
        select_metric=SELECT_METRIC,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        do_cv=bool(DO_CV),
        cv_splits=int(CV_SPLITS),
        cv_scoring=str(CV_SCORING),
    )

    # 导出全量预测 + 保存最佳模型
    export_best_model_predictions(df, label_col, best_name, best_model, drop_cols, out_dir)

    # 保存一次整体元信息（方便复现）
    meta = {
        "input": str(Path(INPUT_FILE).resolve()),
        "label_col": label_col,
        "mode": MODE,
        "select_metric": SELECT_METRIC,
        "drop_cols": drop_cols,
        "text_cols": text_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "n_rows": int(len(df)),
        "positive_rate": float(y.mean()),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

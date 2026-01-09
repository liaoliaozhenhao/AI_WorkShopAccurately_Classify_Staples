import numpy as np
import pandas as pd

def optimize_ab_by_precision_recall_sum(
    df: pd.DataFrame,
    col_label: str = "状态",
    col_x1: str = "LLM_推荐分数",
    col_x2: str = "score_final",
    positive_label: str = "已订",
    r_min: float = 0.0,
    r_max: float = 200.0,
    r_step: float = 0.05,
):
    """
    在 score = A*x1 + B*x2 下，最大化 (准确率 + 查全率)。
    由于整体缩放不影响排序，只搜索 r=B/A，并固定 A=1, B=r。
    对每个 r：按 score 降序排列，扫描 top-K 的切分点，选使 (precision+recall) 最大的 K。
    返回：最优 A,B（给出 A+B=1 的归一化形式）、最佳阈值、最佳K、以及指标。
    """
    # 1) 基础检查
    for c in [col_label, col_x1, col_x2]:
        if c not in df.columns:
            raise ValueError(f"缺少列：{c}")

    y = (df[col_label].astype(str).str.strip() == positive_label).astype(int).to_numpy()
    x1 = df[col_x1].to_numpy(dtype=float)
    x2 = df[col_x2].to_numpy(dtype=float)

    n = len(y)
    pos_total = int(y.sum())
    if pos_total == 0:
        raise ValueError("表格里没有任何“已订”样本，无法优化。")

    def best_for_r(r: float):
        scores = x1 + r * x2
        order = np.argsort(-scores)
        y_sorted = y[order]
        cum_pos = np.cumsum(y_sorted)

        k = np.arange(1, n + 1)
        precision = cum_pos / k
        recall = cum_pos / pos_total
        obj = precision + recall

        best_idx = int(np.argmax(obj))
        best_k = best_idx + 1
        thr = float(scores[order][best_idx])

        return {
            "r": float(r),
            "best_k": int(best_k),
            "threshold_raw": thr,
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "obj": float(obj[best_idx]),
            "pos_in_high": int(cum_pos[best_idx]),
        }

    # 2) 网格搜索 r=B/A
    rs = np.arange(r_min, r_max + 1e-12, r_step, dtype=float)
    best = None
    for r in rs:
        res = best_for_r(float(r))
        if (best is None) or (res["obj"] > best["obj"] + 1e-12):
            best = res

    # 3) 输出 A,B（原始与归一化）
    r_best = best["r"]
    A_raw, B_raw = 1.0, r_best
    A_norm = A_raw / (A_raw + B_raw)
    B_norm = B_raw / (A_raw + B_raw)

    return {
        **best,
        "A_raw": A_raw,
        "B_raw": B_raw,
        "A_norm": A_norm,   # A+B=1
        "B_norm": B_norm,   # A+B=1
    }

def apply_weights_and_save(
    input_path: str,
    output_path: str,
    r_min: float = 0.0,
    r_max: float = 200.0,
    r_step: float = 0.05,
):
    df = pd.read_excel(input_path)

    best = optimize_ab_by_precision_recall_sum(
        df,
        r_min=r_min,
        r_max=r_max,
        r_step=r_step,
    )

    # 使用归一化权重（A+B=1）计算评分
    A = best["A_norm"]
    B = best["B_norm"]

    df_out = df.copy()
    df_out["评分"] = df_out["LLM_推荐分数"].astype(float) * A + df_out["score_final"].astype(float) * B

    # 用“最佳K对应的阈值”来划分高分（注意：阈值会随权重整体缩放，这里是归一化权重下的阈值）
    # 我们重新按归一化评分求一次阈值，保证一致
    scores = df_out["评分"].to_numpy(dtype=float)
    order = np.argsort(-scores)
    thr = float(scores[order][best["best_k"] - 1])

    df_out["是否高分"] = np.where(df_out["评分"] >= thr, "高分", "非高分")

    # 计算最终指标（以该 thr 划分的高分集合为准）
    y = (df_out["状态"].astype(str).str.strip() == "已订").astype(int).to_numpy()
    high = (df_out["评分"] >= thr).to_numpy()
    high_count = int(high.sum())
    pos_total = int(y.sum())
    pos_in_high = int((y[high] == 1).sum())

    precision = pos_in_high / high_count if high_count else 0.0
    recall = pos_in_high / pos_total if pos_total else 0.0

    summary = pd.DataFrame([
        ["样本总数", len(df_out)],
        ["已订数量", pos_total],
        ["未订数量", len(df_out) - pos_total],
        ["最优比值 r=B/A", best["r"]],
        ["A(原始)", best["A_raw"]],
        ["B(原始)", best["B_raw"]],
        ["A(归一化,A+B=1)", A],
        ["B(归一化,A+B=1)", B],
        ["最佳K(理论top-K切分点)", best["best_k"]],
        ["阈值(归一化评分)", thr],
        ["高分数量", high_count],
        ["高分中的已订数量", pos_in_high],
        ["准确率(Precision)", precision],
        ["查全率(Recall)", recall],
        ["准确率+查全率", precision + recall],
    ], columns=["指标", "数值"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        df_out.to_excel(w, index=False, sheet_name="scored")
        summary.to_excel(w, index=False, sheet_name="summary")

    return best, thr, precision, recall

if __name__ == "__main__":
    input_path = "工作簿1.xlsx"
    output_path = "工作簿1_权重优化结果.xlsx"

    best, thr, precision, recall = apply_weights_and_save(
        input_path=input_path,
        output_path=output_path,
        r_min=0.0,
        r_max=200.0,
        r_step=0.05,
    )

    print("=== 最优权重 ===")
    print(f"r=B/A = {best['r']:.4f}")
    print(f"A_norm = {best['A_norm']:.6f}, B_norm = {best['B_norm']:.6f} (A+B=1)")
    print("=== 高分阈值 & 指标 ===")
    print(f"threshold = {thr:.6f}")
    print(f"precision = {precision:.4f}, recall = {recall:.4f}, sum = {precision+recall:.4f}")

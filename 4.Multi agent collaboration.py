import pandas as pd
import re
from typing import Dict, Any
from langchain_community.chat_models import ChatTongyi
import Single_Agent
from policy_rules import (apply_policy_llm_to_df,call_policy_llm,
                          build_book_policy_payload,load_policy_text)
from prefile import (load_collection_profile,infer_call_prefix_from_classno,
                     infer_subject_category_from_classno,add_collection_need_scores)
from Modular_binding import *


# ========== 1. 一些全局常量与工具函数 ==========
SLLM=Single_Agent
# 大类标签（Supervisor 只允许使用这六个）
CATEGORY_NAMES = [
    "人文科学类",
    "社会科学类",
    "理工类",
    "语言教育类",
    "艺术类",
    "其他"
]

# 你可以根据实际表头调整这里，用来给大模型提供“图书描述”
TEXT_COLUMNS = [
    "书名", "题名", "作者", "责任者", "出版社","文献类型","内容简介",
    "ISBN", "分类号", "学科", "内容提要", "简介", "推荐学院"
]

# 只读征订单的加载函数
def load_candidates(full_path: str) -> pd.DataFrame:
    """
    只读取征订单（征订记录合并表.xlsx），不加载订购明细、不打 is_selected 标签。
    用于：仅使用已有模型做选书推理的场景。
    """
    df = pd.read_excel(full_path)
    print("征订单待选书目记录数：", len(df))
    return df


def build_book_description(row: pd.Series) -> str:
    """
    把一行的关键信息拼成一段“图书描述文本”，供大模型判断。
    只使用 TEXT_COLUMNS 中存在于当前表头的列。
    """
    parts = []
    for col in TEXT_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}：{row[col]}")
    if not parts:
        # 如果一个字段都没匹配上，就直接用整行字符串
        return str(row.to_dict())
    return "\n".join(parts)

# ========== 2. 1 号智能体：分类 / 调度 Agent ==========

def classify_category(book_text: str) -> str:
    """
    调用大模型，对图书进行大类分类。
    只允许返回 CATEGORY_NAMES 中的一个。
    """
    system_prompt = """你是山东师范大学图书馆资源建设部的分类员。
你需要根据图书的题名、作者、出版社、分类号、内容简介等信息，
判断该书更适合被归入下面六个大类中的哪一个：
1）人文科学类
2）社会科学类
3）理工类
4）语言教育类
5）艺术类
6）其他

请注意：
- 只能从这六个选项中选一个；
- 回答时只输出类别本身，例如：理工类；
- 不要输出任何多余文字。"""

    user_prompt = f"下面是一本候选图书的信息：\n{book_text}\n\n请给出该书的大类："

    category = SLLM.call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]).strip()

    # 清理一些可能的多余空格
    category = category.replace("：", "").replace(":", "").strip()

    # 防御：如果不在预设列表中，就归为“其他”
    if category not in CATEGORY_NAMES:
        category = "其他"
    return category


# 根据大类，把请求分配到对应的智能体（目前所有大类用同一个评估函数；
# 以后如果你想给每个学科设计不同的策略，可以拆成多个函数）
def route_to_category_agent(book_text: str, category: str) -> Dict[str, Any]:
    """
    对应脑子里的“2～7 号智能体”。
    目前统一调用 evaluate_book_for_category，
    日后可以根据 category 拆成多个不同实现。
    """
    return SLLM.evaluate_book_for_category(book_text, category)

# ========== 4. 主流程：逐条读取 Excel、调度、写回结果 ==========

def process_excel(
    input_path: str = "征订记录合并表.xlsx",
    output_path: str = "征订记录合并表_智能推荐结果.xlsx",
    summary_path: str = "采选统计汇总.xlsx"
):
    # 读取原始征订单
    df = pd.read_excel(input_path)

    # 增加几列：一级分类 / 采纳建议 / 推荐分数 / 推荐理由
    df["一级分类"] = ""
    df["采纳建议"] = ""      # 接受 / 拒绝
    df["LLM_推荐分数"] = 0.0
    df["推荐理由"] = ""

    total = len(df)
    print(f"共读取到 {total} 条记录，开始逐条处理……")

    for idx, row in df.iterrows():
        # 1）构造图书描述（供 1 号智能体分类 & 2～7 号智能体决策使用）
        book_text = build_book_description(row)

        # 2）1 号智能体：判断所属大类
        category = classify_category(book_text)

        # 3）路由到对应的 2～7 号智能体
        result = route_to_category_agent(book_text, category)

        # 4）把结果写回当前行
        df.at[idx, "一级分类"] = category
        df.at[idx, "采纳建议"] = "接受" if result["accept"] else "拒绝"
        df.at[idx, "LLM_推荐分数"] = result["score"]
        df.at[idx, "推荐理由"] = result["reason"]

        print("\r完成进度{0}%".format((idx + 1) * 100 / total), end="", flush=True)
        # if (idx + 1) % 10 == 0 or idx == total - 1:
        #     print(f"已处理 {idx + 1}/{total} 条记录……")

    # 5）保存带有智能推荐结果的新 Excel
    df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存到：{output_path}")

    # 6）顺带做一个按学科的大致汇总统计
    save_summary(df, summary_path)


# ========== 5. 汇总统计（类似 Supervisor 的“收尾工作”） ==========

def save_summary(df: pd.DataFrame, summary_path: str):
    """
    按一级分类做一个简单的统计汇总：
    - 总书目数
    - 采纳数
    - 平均推荐分数
    """
    if "一级分类" not in df.columns or "采纳建议" not in df.columns or "LLM_推荐分数" not in df.columns:
        print("缺少必要字段，无法生成汇总统计。")
        return

    summary = (
        df
        .groupby("一级分类")
        .agg(
            总书目数=("LLM_推荐分数", "size"),
            采纳数=("采纳建议", lambda x: (x == "接受").sum()),
            平均推荐分数=("LLM_推荐分数", "mean")
        )
        .reset_index()
    )

    summary.to_excel(summary_path, index=False)
    print(f"汇总统计已保存到：{summary_path}")
    print(summary)

# ========== 6. 主流程：逐条读取 Excel、调度(LLM+Police+馆藏画像)、写回结果 ==========
def main_infer_with_policy(
    FULL_PATH: str = "征订记录合并表.xlsx",
    COLLECTION_PROFILE_PATH: str = "馆藏与借阅画像分析报告.xlsx",
    POLICY_PATH: str = "policy_rules.txt",
    MODEL_PATH = "book_selector_online.pkl",
    LLM_MODEL_NAME:str = "deepseek"
):
    """
    使用：
    - 已训练好的书目模型 book_selector_online.pkl
    - 馆藏与借阅画像分析报告
    - policy_rules.txt + LLM

    对当前征订单进行选书模拟，并输出带“政策决策”的结果。
    """
    # 1. 只读征订单
    df = load_candidates(FULL_PATH)

    # 2. 应用人工硬规则/软规则
    df = apply_all_rules(df)

    # 3. 加入馆藏/借阅画像的需求分
    profile = load_collection_profile(COLLECTION_PROFILE_PATH)
    df = add_collection_need_scores(df, profile)

    # 4. 构造文本字段 & 向量化
    df = build_text_all(df)
    vectorizer = build_vectorizer()
    X_all = vectorizer.transform(df["text_all"])

    # 5. 加载已有模型
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到已训练模型文件：{MODEL_PATH}，请先运行 main() 训练一次。"
        )
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    print(f"已从 {MODEL_PATH} 加载模型，用于本次带政策的选书推理。")

    # 6. 模型打分 + 规则分（和原来 score_and_recommend 中逻辑一致）
    df["score_model"] = clf.predict_proba(X_all)[:, 1]
    df["score_final"] = (
        df["score_model"]
        + 0.2 * df["rule_pos_score"]
        - 0.1 * df["rule_softneg_score"]
    )

    # 如需加上馆藏需求分，可以改成：
    df["score_final"] = df["score_final"] + 0.2 * df["need_overall_score"]

    # 先根据 score_final + 硬规则做一轮推荐标记
    df["is_recommended"] = (
        (df["rule_hard_reject"] == 0) &
        (df["score_final"] >= RECOMMEND_THRESHOLD)
    ).astype(int)

    # 7. 调用 LLM 做政策评估（只挑前 top_k 本）
    policy_text = load_policy_text(POLICY_PATH)
    df_policy = apply_policy_llm_to_df(
        df, policy_text, top_k=200, model=LLM_MODEL_NAME
    )

    # 8. 输出结果
    #   这里直接复用 save_results，也可以单独写一个，避免误用“命中率”那部分
    dummy_model_metrics = {"说明": "本次仅做推理和政策评估，无训练评估指标。"}
    dummy_rec_metrics = {"说明": "本次未使用订购明细，不计算命中率/查全率。"}

    # 推荐子表：按 policy_decision_level 过滤，如只看“采/建议采”的
    df_rec_policy = df_policy[
        df_policy["policy_decision_level"].isin(["采", "建议采"])
    ].copy()

    save_results(
        df_policy,
        df_rec_policy,
        dummy_model_metrics,
        dummy_rec_metrics,
        out_path="模型选书结果_征订单+政策LLM.xlsx",
    )


# ========== 6. main 入口 ==========

if __name__ == "__main__":


    process_excel(
        # input_path="征订记录合并表-副本.xlsx",
        input_path="征订记录合并表(3000精简测试版).xlsx",
        # input_path="征订记录合并表 - 副本.xlsx",
        output_path="单模型征订记录合并表_智能推荐结果.xlsx",
        summary_path="采选统计汇总.xlsx"
    )

    # 输入建议用你“上一阶段”跑完的表：
    # - 如果你先跑了 Step1（学科LLM）再跑 5.agent_evaluation.py（ML+规则融合）
    #   那这里可以直接喂 “模型选书结果_仅推理_征订单.xlsx”
    # - 或者直接喂 “单模型征订记录合并表_智能推荐结果(精简测试版_0).xlsx”
    # INPUT_EXCEL = "单模型征订记录合并表_智能推荐结果(精简测试版_0).xlsx"
    INPUT_EXCEL = "征订记录合并表_1046.xlsx"

    # run_pipeline(
    #     input_excel=INPUT_EXCEL,
    #     step1_out="step1_LLM学科选书.xlsx",
    #     step2_out="step2_LLM政策审查.xlsx",
    #     step3_out="step3_画像配册配种_最终.xlsx",
    #     step3_summary="step3_画像配册配种_汇总.xlsx",
    #     policy_path="policy_rules.txt",
    #     profile_path="馆藏与借阅画像_profile.json",
    #     auto_skip_step1=True,  # 如果 INPUT_EXCEL 已经有 Step1 结果就会跳过
    #     policy_top_k=50,  # 可先设 200 做测试
    #     total_kind_budget=800  # 若要裁剪总种数，比如 800，就填 800
    # )
    # main_infer_with_policy(
    #     FULL_PATH = "征订记录合并表-副本.xlsx",
    #     COLLECTION_PROFILE_PATH = "馆藏与借阅画像分析报告.xlsx",
    #     POLICY_PATH = "policy_rules.txt",
    #     MODEL_PATH = "book_selector_online.pkl",
    #     LLM_MODEL_NAME = "deepseek"
    # )

    pass


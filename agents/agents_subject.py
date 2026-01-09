#Step1 学科智能体

# agents_subject.py
import pandas as pd
import Single_Agent
from agents.text_utils import build_book_text

CATEGORY_NAMES = ["人文科学类", "社会科学类", "理工类", "语言教育类", "艺术类", "其他类"]

def _norm_category(cat: str) -> str:
    if not isinstance(cat, str):
        return "其他类"
    cat = cat.strip().replace("：", "").replace(":", "")
    if cat == "其他":
        cat = "其他类"
    if cat not in CATEGORY_NAMES:
        cat = "其他类"
    return cat

def classify_category_llm(book_text: str) -> str:
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
    user_prompt = f"候选图书信息如下：\n{book_text}\n\n只输出所属大类："
    cat = Single_Agent.call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    return _norm_category(str(cat))

def run_subject_agent(df: pd.DataFrame, out_path: str, do_classify: bool = True) -> pd.DataFrame:
    df = df.copy()

    # 这些列名尽量与现有生态对齐（方便后续 5.agent_evaluation.py 等脚本复用）
    if "一级分类" not in df.columns:
        df["一级分类"] = ""
    if "采纳建议" not in df.columns:
        df["采纳建议"] = ""
    if "LLM_推荐分数" not in df.columns:
        df["LLM_推荐分数"] = 0.0
    if "推荐理由" not in df.columns:
        df["推荐理由"] = ""

    total = len(df)
    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        book_text = build_book_text(row)

        cat = row.get("一级分类", "")
        if do_classify:
            if not isinstance(cat, str) or not cat.strip():
                cat = classify_category_llm(book_text)
        else:
            cat = _norm_category(cat)

        result = Single_Agent.evaluate_book_for_category(book_text, cat)

        df.at[idx, "一级分类"] = cat
        df.at[idx, "采纳建议"] = "接受" if result.get("accept") else "拒绝"
        df.at[idx, "LLM_推荐分数"] = float(result.get("score", 0.0))
        df.at[idx, "推荐理由"] = result.get("reason", "")

        if i % 20 == 0 or i == total:
            print(f"\r[Agent1-学科] {i}/{total}", end="", flush=True)
    print()

    df.to_excel(out_path, index=False)
    print(f"[Agent1-学科] 输出：{out_path}")
    return df

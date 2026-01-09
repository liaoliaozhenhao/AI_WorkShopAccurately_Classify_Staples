# 把“真实征订目录”字段拼成 LLM 友好文本
import pandas as pd
from typing import List, Optional

DEFAULT_FIELDS = [
    "题名", "副题名", "丛书题名", "分册名", "并列题名", "正题名",
    "责任者", "第二责任者",
    "出版社", "出版年",
    "文献类型", "作品语种", "载体",
    "分类号", "ISBN",
    "主题", "读者对象",
    "内容简介",
    "尺寸", "页码", "版次",
    "价格", "销售价格", "币种",
    "备注", "推荐老师", "推荐者",
    # 如果你输入表里已经有 ML/规则/综合分，存在就自动拼进去
    "score_final_adj", "score_final", "score_model", "馆藏指数",
]

def build_book_text(row: pd.Series, extra_fields: Optional[List[str]] = None) -> str:
    fields = DEFAULT_FIELDS + (extra_fields or [])
    parts = []
    for col in fields:
        if col in row.index and pd.notna(row[col]):
            parts.append(f"{col}：{row[col]}")
    if not parts:
        return str(row.to_dict())
    return "\n".join(parts)

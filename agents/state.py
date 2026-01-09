# state.py
from typing import TypedDict, Dict, Any, Optional
import pandas as pd

class TaskState(TypedDict, total=False):
    # 输入输出
    input_path: str
    output_dir: str
    outputs: Dict[str, str]

    # 配置
    config: Dict[str, Any]

    # 共享数据
    df_raw: pd.DataFrame                  # 原始征订目录（含馆藏重复的全部）
    df_raw_eval: pd.DataFrame             # 去掉馆藏重复后、进入评判的征订目录
    df_holdings_dup: pd.DataFrame         # 馆藏重复清单（直接拦截）

    df_subject: pd.DataFrame
    df_policy: pd.DataFrame
    df_final: pd.DataFrame

    # 共享知识
    policy_text: str
    profile: Dict[str, Any]

    # 流程控制
    halt: bool
    halt_reason: str

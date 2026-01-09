# 运行入口
from sqlalchemy import nullslast, Nullable

from agents.state import TaskState
from pathlib import Path
from agents.graph_pipeline import build_graph_or_fallback
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


def main(state: TaskState=None):
    runner, is_langgraph = build_graph_or_fallback()
    print(f"运行模式：{'LangGraph' if is_langgraph else '顺序执行(无LangGraph)'}")

    if is_langgraph:
        final = runner.invoke(state)   # LangGraph
        if final.get("halt", False):
            print(final.get("halt_reason", "流程已暂停"))
            return
    else:
        final = runner(state)          # fallback

    print("\n=== 输出文件清单 ===")
    for k, v in final.get("outputs", {}).items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    SUBJECT_POLICY_DIR = Path("policy")
    policy_path = SUBJECT_POLICY_DIR / "policy_rules.txt"
    state: TaskState = {
        "input_path": "征订记录合并表(1000精简测试版).xlsx",
        # "input_path": "征订记录合并表(精简).xlsx",
        "output_dir": "outputs_1000_test",
        "outputs": {},
        "config": {
            "policy_path": policy_path,
            "profile_path": "馆藏与借阅画像_profile.json",
            "holdings_path": "馆藏清单-按种20260106-005.xlsx",

            # Step2 政策审查范围控制：建议先测试 top_k=100 或 200
            "policy_scope": "accepted_only",
            "policy_top_k": None,  # None=全部学科接受的都审查；可改为 200

            # Step3 配种预算：None=不裁剪；如果想控制总种数，比如 600，就填 600
            "total_kind_budget": None,
            "per_category_min": 0,
            "per_category_max": None,

            "enable_subject_calibration": True,
            "calibration_sample_n": 20,  # 10-30
            "seed": 42,

            # 人工反馈的文件值设置
            "feedback_mode": "soft",  # 推荐 soft  有 soft / off / full
            "feedback_min_samples": 12,  # 小样本太少不启用纠偏细则   少于该值就不启用（只写弱提示）
            "feedback_max_items": 8,  # 典型备注最多保留8条  最多保留多少条典型备注
        }
    }
    main(state)

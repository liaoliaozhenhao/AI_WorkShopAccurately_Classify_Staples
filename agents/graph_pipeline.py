#总控：LangGraph 串联

from typing import Dict, Any
import numpy as np
from agents.state import TaskState
from agents.io_utils import ensure_dir, read_excel, write_excel_sheets
from agents.agents_subject import run_subject_agent
from agents.agents_policy import load_policy_text, run_policy_agent
from agents.agents_allocation import load_profile, allocate_kinds_and_copies
import os
import pandas as pd
from agents.agents_holdings_dedupe import build_holdings_index, mark_and_split_dedupe
from agents.subject_feedback_loop import (
    sample_for_calibration, export_review_sheet, load_human_feedback,
    build_feedback_notes, write_feedback_file, auto_fill_feedback_remarks
)

def node_load(state: TaskState) -> Dict[str, Any]:
    input_path = state["input_path"]
    out_dir = ensure_dir(state["output_dir"])
    df_raw = read_excel(input_path)
    outputs = dict(state.get("outputs", {}))

    df_raw = df_raw.reset_index(drop=True)
    df_raw["_row_id"] = df_raw.index

    return {"df_raw": df_raw, "outputs": outputs, "output_dir": out_dir}

def node_subject(state: TaskState) -> Dict[str, Any]:
    df = state.get("df_raw_eval", state["df_raw"])
    out_dir = state["output_dir"]
    step1_path = f"{out_dir}/step1_LLM学科选书.xlsx"
    df_subject = run_subject_agent(df, out_path=step1_path, do_classify=True)

    outputs = dict(state.get("outputs", {}))
    outputs["step1_subject"] = step1_path
    return {"df_subject": df_subject, "outputs": outputs}

def node_policy(state: TaskState) -> Dict[str, Any]:
    if "df_subject" not in state:
        return {
            "halt": True,
            "halt_reason": "未找到 df_subject：学科全量阶段未执行或流程未正确 halt。请检查 build_graph_or_fallback 的连接顺序。"
        }

    cfg = state.get("config", {})
    out_dir = state["output_dir"]

    policy_path = cfg.get("policy_path", r"Accurately_Classify_Staples\policy_rules.txt")
    policy_text = load_policy_text(policy_path)

    step2_path = f"{out_dir}/step2_LLM政策审查.xlsx"
    df_policy = run_policy_agent(
        state["df_subject"],
        policy_text=policy_text,
        out_path=step2_path,
        scope=cfg.get("policy_scope", "accepted_only"),
        top_k=cfg.get("policy_top_k", None),
    )

    outputs = dict(state.get("outputs", {}))
    outputs["step2_policy"] = step2_path
    return {"policy_text": policy_text, "df_policy": df_policy, "outputs": outputs}

def node_allocation(state: TaskState) -> Dict[str, Any]:
    cfg = state.get("config", {})
    out_dir = state["output_dir"]

    profile_path = cfg.get("profile_path", "馆藏与借阅画像_profile.json")
    profile = load_profile(profile_path)

    step3_path = f"{out_dir}/step3_画像配种配册_全表.xlsx"
    step3_sum = f"{out_dir}/step3_画像配种配册_汇总.xlsx"

    df_final, df_sum = allocate_kinds_and_copies(
        state["df_policy"],
        profile=profile,
        out_path=step3_path,
        summary_path=step3_sum,
        total_kind_budget=cfg.get("total_kind_budget", None),
        per_category_min=cfg.get("per_category_min", 0),
        per_category_max=cfg.get("per_category_max", None),
    )

    outputs = dict(state.get("outputs", {}))
    outputs["step3_final"] = step3_path
    outputs["step3_summary"] = step3_sum
    return {"profile": profile, "df_final": df_final, "outputs": outputs}

def temp_decision_by_subject_and_profile(df):
    """
    对政策=待审查的条目，用 学科 + 画像 给出临时建议：
    - 采纳建议=拒绝 -> 临时不采
    - 否则综合分 = 0.6*学科分 + 0.4*画像分 （都压到0~1）
    """
    d = df.copy()

    # 学科分：LLM_推荐分数通常在[-1,1]，映射到[0,1]
    llm = d["LLM_推荐分数"].astype(float).fillna(0.0)
    llm01 = (llm + 1.0) / 2.0
    prof = d["画像需求分"].astype(float).fillna(0.0)

    score = 0.6 * llm01 + 0.4 * prof
    d["临时综合分"] = score

    d["临时建议"] = "临时复核"
    d.loc[(d["采纳建议"] == "拒绝"), "临时建议"] = "临时不采"
    d.loc[(d["采纳建议"] == "接受") & (score >= 0.70), "临时建议"] = "临时推荐"
    d.loc[(d["采纳建议"] == "接受") & (score < 0.55), "临时建议"] = "临时不采"

    return d


def node_export_summary(state):
    out_dir = state["output_dir"]
    df = state["df_final"].copy()

    # ---- 缺列兜底 ----
    for col, default in [
        ("final_selected", 0),
        ("采纳建议", ""),
        ("政策决策", ""),
        ("政策放行", np.nan),   # 重要：待审查时保持 NaN，不要用 0
        ("LLM_推荐分数", 0.0),
        ("画像需求分", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    # ---- 定义政策状态：audited / pending ----
    policy_pending = df["政策决策"].isna() | (df["政策决策"].astype(str).str.strip() == "") | (df["政策决策"] == "待审查")
    policy_audited = ~policy_pending

    # ---- 对“待审查”生成临时建议：学科+画像 ----
    pending = df[policy_pending].copy()
    if len(pending) > 0:
        llm = pending["LLM_推荐分数"].astype(float).fillna(0.0)
        llm01 = (llm + 1.0) / 2.0
        prof = pending["画像需求分"].astype(float).fillna(0.0)
        pending["临时综合分"] = 0.6 * llm01 + 0.4 * prof

        pending["临时建议"] = "临时复核"
        pending.loc[(pending["采纳建议"] == "拒绝"), "临时建议"] = "临时不采"
        pending.loc[(pending["采纳建议"] == "接受") & (pending["临时综合分"] >= 0.70), "临时建议"] = "临时推荐"
        pending.loc[(pending["采纳建议"] == "接受") & (pending["临时综合分"] < 0.55), "临时建议"] = "临时不采"

        # 回写到 df
        df.loc[pending.index, "临时综合分"] = pending["临时综合分"]
        df.loc[pending.index, "临时建议"] = pending["临时建议"]
    else:
        df["临时综合分"] = np.nan
        df["临时建议"] = ""

    # ---- 先打 bucket（互斥！）----
    df["bucket"] = "UNASSIGNED"

    # ✅ 人工直推：无条件进最终推荐（仅限政策已审查或你认为可直推的场景）
    if "人工直推" in df.columns:
        df.loc[policy_audited & (df["人工直推"] == 1), "bucket"] = "01_最终推荐清单"

    # 1) 待政策审查（优先拦截，避免被“不采”误伤）
    df.loc[policy_pending & (df["临时建议"] == "临时推荐"), "bucket"] = "04_待政策审查_临时推荐"
    df.loc[policy_pending & (df["临时建议"] == "临时复核"), "bucket"] = "05_待政策审查_临时复核"
    df.loc[policy_pending & (df["临时建议"] == "临时不采"), "bucket"] = "06_待政策审查_临时不采"

    # 2) 政策已审查：最终推荐（只放 audited，pending 的推荐留在 04）
    df.loc[policy_audited & (df["final_selected"] == 1), "bucket"] = "01_最终推荐清单"

    # 3) 政策已审查：人工复核
    review_cond = policy_audited & (
        (df["政策决策"] == "需人工复核") |
        ((df["采纳建议"] == "拒绝") & (df["政策放行"] == 1))
    )
    review_cond = review_cond | (df.get("人工需复核", 0) == 1)
    df.loc[review_cond & (df["bucket"] == "UNASSIGNED"), "bucket"] = "02_人工复核清单"

    # 4) 政策已审查：不采（注意排除前面已经归类的）
    reject_cond = policy_audited & (
        (df["采纳建议"] == "拒绝") |
        (df["政策放行"] == 0)
    )
    df.loc[reject_cond & (df["bucket"] == "UNASSIGNED"), "bucket"] = "03_不采清单"

    # 5) 其它：合格但未入选（可选）
    eligible_not_selected = policy_audited & (df["采纳建议"] == "接受") & (df["政策放行"] == 1) & (df["final_selected"] != 1)
    df.loc[eligible_not_selected & (df["bucket"] == "UNASSIGNED"), "bucket"] = "08_合格但未入选"

    # 6) 最后兜底（理论上不会有）
    df.loc[df["bucket"] == "UNASSIGNED", "bucket"] = "09_未归类_请检查"

    df.loc[df.get("人工直推", 0) == 1, "bucket"] = "01_最终推荐清单"
    df_dup = state.get("df_holdings_dup")
    if df_dup is None:
        df_dup = pd.DataFrame()

    df_dup = state.get("df_holdings_dup", pd.DataFrame())
    # ---- 输出各 sheet（互斥，行数之和 = len(df)）----
    sheets = {
        "00_馆藏重复清单": df_dup.copy() if isinstance(df_dup, pd.DataFrame) else pd.DataFrame(),
        "01_最终推荐清单": df[df["bucket"] == "01_最终推荐清单"].copy(),
        "02_人工复核清单": df[df["bucket"] == "02_人工复核清单"].copy(),
        "03_不采清单": df[df["bucket"] == "03_不采清单"].copy(),
        "04_待政策审查_临时推荐": df[df["bucket"] == "04_待政策审查_临时推荐"].copy(),
        "05_待政策审查_临时复核": df[df["bucket"] == "05_待政策审查_临时复核"].copy(),
        "06_待政策审查_临时不采": df[df["bucket"] == "06_待政策审查_临时不采"].copy(),
        "08_合格但未入选": df[df["bucket"] == "08_合格但未入选"].copy(),
        "07_全量结果": df.copy(),
    }



    final_path = f"{out_dir}/汇总智能推荐表.xlsx"
    write_excel_sheets(final_path, sheets)

    # 打印核对信息（建议保留）
    total = len(df)
    parts = sum(len(sheets[k]) for k in sheets if k not in ("07_全量结果", "00_馆藏重复清单"))
    print(f"[汇总校验] 全量={total}；各分表合计(不含全量)={parts}")
    if parts != total:
        print("[警告] 分表合计 != 全量，说明仍存在未归类或重复导出问题（检查 bucket 逻辑）")

    outputs = dict(state.get("outputs", {}))
    outputs["final_summary"] = final_path
    return {"outputs": outputs}


def build_graph_or_fallback():
    """
    优先使用 LangGraph；若环境无 LangGraph，则返回顺序执行函数。
    """
    try:
        from langgraph.graph import StateGraph, END

        def route_if_halt(state: TaskState) -> str:
            return "halt" if state.get("halt", False) else "continue"

        graph = StateGraph(TaskState)
        graph.add_node("load", node_load)
        #馆藏去重节点
        graph.add_node("holdings_dedupe", node_holdings_dedupe)

        # ✅ 两段式学科节点
        graph.add_node("subject_calibration", node_subject_calibration)
        graph.add_node("subject_full", node_subject_full)

        graph.add_node("policy", node_policy)
        graph.add_node("allocation", node_allocation)
        graph.add_node("export", node_export_summary)

        #开始
        graph.set_entry_point("load")
        #开始 到 馆藏去重
        graph.add_edge("load", "holdings_dedupe")
        #馆藏去重 到 学科校准
        graph.add_edge("holdings_dedupe", "subject_calibration")

        # ✅ 若校准阶段 halt（第一次运行）则直接结束
        graph.add_conditional_edges(
            "subject_calibration",
            route_if_halt,
            {"halt": END, "continue": "subject_full"}
        )

        # ✅ 若全量阶段 halt（反馈表未填）也直接结束
        graph.add_conditional_edges(
            "subject_full",
            route_if_halt,
            {"halt": END, "continue": "policy"}
        )

        # graph.add_edge("subject_calibration", "subject_full")
        # graph.add_edge("subject_full", "policy")

        graph.add_edge("policy", "allocation")
        graph.add_edge("allocation", "export")
        graph.add_edge("export", END)

        return graph.compile(), True
    except Exception:
        # fallback：顺序执行
        def run_seq(state: TaskState) -> TaskState:

            upd = node_load(state);state.update(upd)
            upd = node_holdings_dedupe(state);state.update(upd)

            # ✅ 先校准
            upd = node_subject_calibration(state);state.update(upd)
            if state.get("halt", False):
                return state

            # ✅ 再全量（会写入反馈文件）
            upd = node_subject_full(state);
            state.update(upd)
            if state.get("halt", False):
                return state

            upd = node_policy(state);state.update(upd)
            upd = node_allocation(state);state.update(upd)
            upd = node_export_summary(state);state.update(upd)
            return state

        return run_seq, False


def node_subject_calibration(state):
    cfg = state.get("config", {})
    if not cfg.get("enable_subject_calibration", False):
        return {"halt": False}  # ✅ 明确不halt

    out_dir = state["output_dir"]
    review_path = f"{out_dir}/step1_学科校准_人工反馈.xlsx"

    if not os.path.exists(review_path):
        sample_n = int(cfg.get("calibration_sample_n", 20))
        base_df = state.get("df_raw_eval", state["df_raw"])
        sample_df = sample_for_calibration(base_df, n=sample_n, seed=int(cfg.get("seed", 42)))

        # 校准阶段建议先跑“无反馈基线”，避免旧反馈污染样本打分
        try:
            feedback_file = os.path.join("subject_policies", "_human_feedback_global.txt")
            os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
            write_feedback_file(build_feedback_notes(pd.DataFrame(), mode="off"), feedback_file)
        except Exception:
            pass

        sample_scored = run_subject_agent(sample_df, out_path=f"{out_dir}/step1_学科校准_样本打分.xlsx", do_classify=True)
        export_review_sheet(sample_scored, review_path)

        return {"halt": True, "halt_reason": f"已生成学科校准反馈表：{review_path}。请填写“人工判定/人工备注”后重新运行。"}

    return {"halt": False}  # ✅ 反馈表已存在，继续
def node_subject_full(state: TaskState) -> Dict[str, Any]:
    """
    学科智能体全量挑书（支持“先小样本→人工反馈→再全量”闭环）：

    1) 若启用校准：
       - 读取 output_dir/step1_学科校准_人工反馈.xlsx
       - 自动补全人工备注（生成 _自动补备注.xlsx）
       - 用补备注后的反馈生成“纠偏要点”写入 subject_policies/_human_feedback_global.txt
    2) 对（去重后的）df_raw_eval 进行全量学科评判，输出 step1_LLM学科选书.xlsx
    3) 将反馈样例按人工意见“硬覆盖”（样例结果以人工为准）：
       - 人工判定=接受 => 采纳建议=接受，人工直推=1
       - 人工判定=拒绝 => 采纳建议=拒绝
       - 人工判定=复核 => 采纳建议=接受，但人工需复核=1
       - 人工一级分类非空 => 覆盖一级分类
       - 人工备注合入“推荐理由”
    4) 返回 df_subject 给后续 policy/allocation/export 使用
    """

    cfg = state.get("config", {}) or {}
    out_dir = state["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    enable_calib = bool(cfg.get("enable_subject_calibration", False))
    review_path = os.path.join(out_dir, "step1_学科校准_人工反馈.xlsx")

    fb = None

    # ---------- 0) 读取人工反馈 + 自动补备注 + 写入纠偏要点 ----------
    if enable_calib:
        # 若还没生成/没填反馈表，则暂停（正常流程应由 node_subject_calibration 先生成并 halt）
        if not os.path.exists(review_path):
            return {
                "halt": True,
                "halt_reason": f"未找到人工反馈表：{review_path}。请先运行校准阶段生成该表并填写“人工判定”。",
            }

        # 自动补备注（并将此文件作为“本轮学习与覆盖”的来源）
        filled_path = os.path.join(out_dir, "step1_学科校准_人工反馈_自动补备注.xlsx")
        auto_fill_feedback_remarks(review_path, save_path=filled_path)

        fb = load_human_feedback(filled_path)
        if fb is None or len(fb) == 0:
            return {
                "halt": True,
                "halt_reason": f"反馈表未填写有效“人工判定”（接受/拒绝/复核）。请先填写：{review_path}",
            }

        # 生成纠偏要点（写入全局反馈文件供学科智能体提示词加载）
        fb_mode = cfg.get("feedback_mode", "soft")  # soft / off
        fb_min_samples = int(cfg.get("feedback_min_samples", 12))
        fb_max_items = int(cfg.get("feedback_max_items", 8))
        notes = build_feedback_notes(
            fb,
            mode=fb_mode,
            min_samples=fb_min_samples,
            max_items=fb_max_items,
        )
        feedback_file = os.path.join("subject_policies", "_human_feedback_global.txt")
        os.makedirs(os.path.dirname(feedback_file), exist_ok=True)
        write_feedback_file(notes, feedback_file)


        outputs = dict(state.get("outputs", {}))
        outputs["step1_feedback_review"] = review_path
        outputs["step1_feedback_filled"] = filled_path
        outputs["step1_feedback_rules"] = feedback_file
        state["outputs"] = outputs

    # ---------- 1) 全量学科评判（对去重后的 df_raw_eval 评判；没有则退回 df_raw） ----------
    base_df = state.get("df_raw_eval", state["df_raw"])
    if "_row_id" not in base_df.columns:
        base_df = base_df.reset_index(drop=True).copy()
        base_df["_row_id"] = base_df.index

    step1_path = os.path.join(out_dir, "step1_LLM学科选书.xlsx")
    df_subject = run_subject_agent(base_df, out_path=step1_path, do_classify=True)

    # ---------- 2) 样例按人工意见硬覆盖（确保“人工接受=直推”） ----------
    if enable_calib and fb is not None and len(fb) > 0:
        df_subject = apply_human_overrides(df_subject, fb)

        # 覆盖后重新保存 step1（确保样例在结果里“按人工意见”）
        df_subject.to_excel(step1_path, index=False)

    # ---------- 3) 回写 outputs + 返回给后续节点 ----------
    outputs = dict(state.get("outputs", {}))
    outputs["step1_subject"] = step1_path

    return {
        "df_subject": df_subject,
        "outputs": outputs,
        "halt": False,
    }

def apply_human_overrides(df_subject: pd.DataFrame, fb: pd.DataFrame) -> pd.DataFrame:
    df = df_subject.copy()

    # 确保有 _row_id
    if "_row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["_row_id"] = df.index

    # 只保留需要的反馈列
    cols = ["_row_id", "人工判定", "人工一级分类", "人工备注"]
    for c in cols:
        if c not in fb.columns:
            fb[c] = ""
    fb2 = fb[cols].copy()
    fb2["人工判定"] = fb2["人工判定"].astype(str).str.strip()
    fb2["人工一级分类"] = fb2["人工一级分类"].astype(str).str.strip()

    # merge
    df = df.merge(fb2, on="_row_id", how="left")

    # 标记人工覆盖
    df["人工覆盖"] = 0
    mask = df["人工判定"].isin(["接受", "拒绝", "复核"])
    df.loc[mask, "人工覆盖"] = 1

    # 复核：为了让后续政策/画像能继续跑，这里把“复核”映射为“接受”，并加标记
    df["人工需复核"] = 0
    df.loc[mask & (df["人工判定"] == "复核"), "人工需复核"] = 1

    # 覆盖一级分类（只有你填写了才覆盖）
    m_cat = mask & (df["人工一级分类"].notna()) & (df["人工一级分类"].astype(str).str.strip() != "")
    df.loc[m_cat, "一级分类"] = df.loc[m_cat, "人工一级分类"]

    # 覆盖采纳建议
    map_dec = {"接受": "接受", "拒绝": "拒绝", "复核": "接受"}
    df.loc[mask, "采纳建议"] = df.loc[mask, "人工判定"].map(map_dec)

    # 把人工备注拼到理由里（可选，但建议）
    if "推荐理由" not in df.columns:
        df["推荐理由"] = ""
    df.loc[mask, "推荐理由"] = (
        df.loc[mask, "推荐理由"].fillna("").astype(str)
        + " | 人工备注：" + df.loc[mask, "人工备注"].fillna("").astype(str)
    )
    # ✅ 人工接受 => 后续直接走“最终推荐”（通过 人工直推 贯穿 policy/allocation/export）
    if "人工直推" not in df.columns:
        df["人工直推"] = 0
    df.loc[mask & (df["人工判定"] == "接受"), "人工直推"] = 1

    return df


def node_holdings_dedupe(state):
    cfg = state.get("config", {})
    out_dir = state["output_dir"]

    holdings_path = cfg.get("holdings_path", "")
    if not holdings_path:
        # 未配置馆藏文件，直接不去重
        return {"df_raw_eval": state["df_raw"], "df_holdings_dup": pd.DataFrame()}

    # 建议用 read_excel（你项目里已有），也可用 pd.read_excel
    df_hold = read_excel(holdings_path,header=4)

    # ✅ 新接口：只返回 1 个 holdings_index（dict）
    holdings_index = build_holdings_index(df_hold)

    # ✅ 新接口：只传 2 个参数 (df_raw, holdings_index)
    df_eval, df_dup = mark_and_split_dedupe(state["df_raw"], holdings_index)

    # 输出中间表，便于核对
    df_dup.to_excel(f"{out_dir}/step0_馆藏重复清单.xlsx", index=False)
    df_eval.to_excel(f"{out_dir}/step0_可评判清单.xlsx", index=False)

    return {"df_raw_eval": df_eval, "df_holdings_dup": df_dup}
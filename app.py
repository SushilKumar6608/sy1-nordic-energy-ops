"""Streamlit review queue for IRREVERSIBLE action approvals."""

import streamlit as st

from src.policy.engine import PolicyEngine, PermissionTier
from src.tools.approval_gate import (
    approve,
    clear_approvals,
    get_pending,
    request_approval,
)
from src.policy.audit import get_audit_log, clear_audit_log

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SY1-Gov - Operator Review Queue",
    page_icon="🔒",
    layout="wide",
)

st.title("🔒 SY1-Gov - Operator Review Queue")
st.caption(
    "Review and approve or reject IRREVERSIBLE actions before they are executed."
)

# ---------------------------------------------------------------------------
# Sidebar - seed demo approvals for testing
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Demo Controls")
    st.markdown("Add sample pending approvals to test the UI.")

    if st.button("Seed: dispatch_maintenance_crew"):
        engine = PolicyEngine()
        decision = engine.evaluate(
            "dispatch_maintenance_crew", {"node_id": "N-42", "priority": "high"}
        )
        request_approval(decision, {"node_id": "N-42", "priority": "high"})
        st.success("Added dispatch_maintenance_crew")

    if st.button("Seed: isolate_grid_segment"):
        engine = PolicyEngine()
        decision = engine.evaluate(
            "isolate_grid_segment", {"segment_id": "S-07", "reason": "fault"}
        )
        request_approval(decision, {"segment_id": "S-07", "reason": "fault"})
        st.success("Added isolate_grid_segment")

    st.divider()
    if st.button("Clear all pending approvals", type="secondary"):
        clear_approvals()
        st.rerun()

    if st.button("Clear audit log", type="secondary"):
        clear_audit_log()
        st.rerun()

# ---------------------------------------------------------------------------
# Pending approvals
# ---------------------------------------------------------------------------
st.subheader("Pending Approvals")

pending = get_pending()

if not pending:
    st.info("No pending approvals. Use the sidebar to seed demo items.")
else:
    for item in pending:
        tier_color = "🔴" if item.decision.tier == PermissionTier.IRREVERSIBLE else "🟡"
        with st.expander(
            f"{tier_color} {item.tool_name}  |  ID: {item.approval_id[:8]}...",
            expanded=True,
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Tool**")
                st.code(item.tool_name)
                st.markdown("**Tier**")
                st.code(item.decision.tier.value)
                st.markdown("**Policy reason**")
                st.write(item.decision.reason)

            with col2:
                st.markdown("**Inputs**")
                st.json(item.inputs)
                st.markdown("**Approval ID**")
                st.code(item.approval_id)

            approve_col, reject_col, _ = st.columns([1, 1, 4])

            with approve_col:
                if st.button(
                    "✅ Approve",
                    key=f"approve_{item.approval_id}",
                    type="primary",
                ):
                    approve(item.approval_id)
                    st.success(f"Approved: {item.tool_name}")
                    st.rerun()

            with reject_col:
                if st.button(
                    "❌ Reject",
                    key=f"reject_{item.approval_id}",
                    type="secondary",
                ):
                    # Rejection = simply remove from pending without approving
                    clear_approvals()
                    st.warning(f"Rejected: {item.tool_name}")
                    st.rerun()

# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Audit Log")

audit = get_audit_log()

if not audit:
    st.info("Audit log is empty.")
else:
    st.markdown(f"**{len(audit)} entries**")

    for entry in reversed(audit):
        allowed = entry.get("allowed", False)
        icon = "✅" if allowed else "🚫"
        tool = entry.get("tool_name", "unknown")
        tier = entry.get("tier", "")
        reason = entry.get("reason", "")
        requires = entry.get("requires_approval", False)

        with st.expander(f"{icon} {tool}  ({tier})", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Allowed:** {allowed}")
                st.markdown(f"**Requires approval:** {requires}")
            with col2:
                st.markdown(f"**Reason:** {reason}")

"""Workflow and Telemetry UI component for Nexus Multi-Agent Research System.

Renders an interactive, comprehensive pipeline trace on the left sidebar:
1. Question & Query Formation (Classification, pronoun resolution, rewriting, decomposition)
2. Document Selection & Scoping (Catalog target, aliases, retrieval mode)
3. Retrieval Data (All retrieved chunks, scores, match types, queries)
4. Top Chunks Selection (Curator ranking, neighbor expansion, sufficiency)
5. Answer Formation (Pre-verification draft vs final answer, citations)
6. Verification & Claims Audit (Fact checks, supported/unsupported claims, strictness)
7. LLM Calls, Token Consumption & Timing Breakdown
"""

from __future__ import annotations

import re
from typing import Any

import streamlit as st


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "0.000s"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _format_tokens(num: int | None) -> str:
    if not num:
        return "0"
    return f"{num:,}"


def render_workflow_inspector(
    workflow: dict[str, Any] | None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> None:
    """Render the full execution workflow and telemetry panel in Streamlit sidebar."""
    st.markdown("### ⚡ Execution Workflow")
    st.caption("Live pipeline trace & telemetry across all agent stages.")

    # 1. Turn Selector if historical workflows exist
    history_workflows: list[tuple[int, str, dict[str, Any]]] = []
    if conversation_history:
        turn_idx = 1
        for msg in conversation_history:
            if msg.get("role") == "assistant" and msg.get("workflow"):
                preview = (msg.get("content") or "")[:40].replace("\n", " ").strip()
                history_workflows.append((turn_idx, f"Turn {turn_idx}: {preview}...", msg["workflow"]))
                turn_idx += 1

    active_workflow = workflow
    if len(history_workflows) > 1:
        options = [item[1] for item in history_workflows]
        default_index = len(options) - 1
        selected_turn_str = st.selectbox(
            "Viewing Query Turn",
            options=options,
            index=default_index,
            key="workflow_turn_selector",
            help="Select which question-answering turn to inspect in detail.",
        )
        for _, label, wf in history_workflows:
            if label == selected_turn_str:
                active_workflow = wf
                break
    elif not active_workflow and history_workflows:
        active_workflow = history_workflows[-1][2]

    # Empty State when no question has run yet
    if not active_workflow:
        st.info(
            "💡 **No workflow active yet.**\n\n"
            "Ask a question in the chat to see real-time:\n"
            "- 🎯 Query rewriting & intent\n"
            "- 📂 Document scoping\n"
            "- 🔍 All retrieved chunks & scores\n"
            "- 🏆 Curator top-chunks selection\n"
            "- ✍️ Answer drafting\n"
            "- 🛡️ Cohere claim verification\n"
            "- ⏱️ LLM tokens & stage timing"
        )
        return

    # Extract workflow structures safely
    plan = active_workflow.get("orchestrator_plan") or {}
    q_info = active_workflow.get("question_type_info") or {}
    verdict = active_workflow.get("verdict") or {}
    token_usage = active_workflow.get("token_usage") or {}
    timings = active_workflow.get("timings") or []
    output_validation = active_workflow.get("output_validation") or {}
    summary_coverage = active_workflow.get("summary_coverage") or {}
    all_chunks = active_workflow.get("chunks")
    legacy_chunks = all_chunks is None
    if legacy_chunks:
        all_chunks = active_workflow.get("evidence")
    if legacy_chunks and not all_chunks:
        all_chunks = (
            active_workflow.get("context_blocks")
            or active_workflow.get("ranked_chunks")
            or active_workflow.get("curated")
        )
    all_chunks = all_chunks or []
    curated_chunks = (
        active_workflow.get("context_blocks")
        or active_workflow.get("ranked_chunks")
        or active_workflow.get("curated")
        or []
    )
    raw_answer = str(active_workflow.get("raw_answer") or active_workflow.get("draft") or "")
    final_answer = str(active_workflow.get("answer") or "")
    citations = list(active_workflow.get("citations") or [])
    total_llm_calls = int(active_workflow.get("llm_calls") or 0)
    total_tokens = int(token_usage.get("total_tokens") or 0)
    prompt_tokens = int(token_usage.get("prompt_tokens") or 0)
    completion_tokens = int(token_usage.get("completion_tokens") or 0)

    query_span = next(
        (
            timing
            for timing in timings
            if timing.get("category") == "QUERY" and timing.get("level") == 0
        ),
        None,
    )
    total_latency_seconds = (
        float(query_span.get("elapsed_seconds", 0.0))
        if query_span
        else 0.0
    )
    if total_latency_seconds <= 0:
        total_latency_seconds = sum(t.get("elapsed_seconds", 0.0) for t in timings)

    # Status badge
    is_ok = verdict.get("ok", True) if verdict else True
    unsupported_count = len([c for c in verdict.get("claims", []) if c.get("supported") != "yes"]) if verdict else 0

    # Summary KPI row
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Latency", _format_seconds(total_latency_seconds))
        st.metric("LLM Calls", f"{total_llm_calls} call(s)")
    with col2:
        st.metric("Token Usage", _format_tokens(total_tokens))
        verif_label = "✅ 100% Verified" if (is_ok and unsupported_count == 0) else f"⚠️ {unsupported_count} Filtered"
        st.metric("Verification", verif_label)

    st.markdown("---")

    # -------------------------------------------------------------
    # STAGE 1: Question & Query Formation
    # -------------------------------------------------------------
    with st.expander("🎯 1. Query Formation & Decomposition", expanded=True):
        raw_q = active_workflow.get("question", "")
        st.markdown(f"**Original User Input:**\n> *{raw_q}*")

        primary_type = q_info.get("primary_type") or plan.get("intent") or "factual"
        action = plan.get("action") or "retrieve"
        ambiguity = plan.get("ambiguity") or "none"

        badges_html = f"""
        <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px;">
            <span style="background: rgba(88, 166, 255, 0.2); color: #58a6ff; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">Type: {primary_type.upper()}</span>
            <span style="background: rgba(63, 185, 80, 0.2); color: #3fb950; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">Action: {action}</span>
            <span style="background: rgba(210, 153, 34, 0.2); color: #d29922; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 600;">Ambiguity: {ambiguity}</span>
        </div>
        """
        st.markdown(badges_html, unsafe_allow_html=True)

        rewritten = plan.get("rewritten_query") or active_workflow.get("reformulated_question") or raw_q
        if rewritten and rewritten.strip().lower() != raw_q.strip().lower():
            st.markdown(f"**Rewritten Query (Self-Contained):**\n`{rewritten}`")

        resolved_q = q_info.get("resolved_question")
        if q_info.get("is_followup") and resolved_q and resolved_q != raw_q:
            st.markdown(f"**Follow-up Pronoun Grounded:**\n`{resolved_q}`")

        if q_info.get("comparison_entities"):
            st.markdown(f"**Extracted Entities:** `{', '.join(q_info['comparison_entities'])}`")

        sub_qs = plan.get("sub_questions") or []
        if sub_qs:
            st.markdown(f"**Sub-Questions Planned ({len(sub_qs)}):**")
            for idx, sq in enumerate(sub_qs, 1):
                sq_query = sq.get("query") if isinstance(sq, dict) else getattr(sq, "query", "")
                sq_mode = sq.get("retrieval") if isinstance(sq, dict) else getattr(sq, "retrieval", "hybrid")
                sq_type = sq.get("question_type") if isinstance(sq, dict) else getattr(sq, "question_type", "")
                st.caption(f"**Q{idx}** `[{sq_mode} · {sq_type}]`: {sq_query}")

    # -------------------------------------------------------------
    # STAGE 2: Document Selection & Scoping
    # -------------------------------------------------------------
    with st.expander("📂 2. Document Selection & Scope", expanded=False):
        doc_ids: list[str] = []
        for sq in plan.get("sub_questions") or []:
            d = sq.get("doc_ids") if isinstance(sq, dict) else getattr(sq, "doc_ids", None)
            if d:
                doc_ids.extend(d)
        doc_ids = list(dict.fromkeys(doc_ids))

        unique_doc_titles = list(dict.fromkeys(
            str(c.get("title") or c.get("source_path") or "")
            for c in all_chunks
            if c.get("title") or c.get("source_path")
        ))

        if doc_ids:
            st.success(f"🎯 **Targeted Scope:** `{', '.join(doc_ids)}`")
        else:
            st.info("🌐 **Targeted Scope:** Global Corpus (All Indexed Documents)")

        if unique_doc_titles:
            st.markdown(f"**Documents Actively Matched ({len(unique_doc_titles)}):**")
            for t in unique_doc_titles[:5]:
                st.markdown(f"- 📄 `{t}`")
            if len(unique_doc_titles) > 5:
                st.caption(f"...and {len(unique_doc_titles) - 5} more")

        r_mode = plan.get("sub_questions", [{}])[0].get("retrieval", "hybrid") if plan.get("sub_questions") else "hybrid"
        st.markdown(f"**Strategy Mode:** `{r_mode}`")
        if plan.get("reason"):
            st.caption(f"**Routing Reason:** {plan.get('reason')}")

    # -------------------------------------------------------------
    # STAGE 3: Retrieval Data (All Retrieved Chunks)
    # -------------------------------------------------------------
    with st.expander(f"🔍 3. Retrieval Data ({len(all_chunks)} chunks)", expanded=False):
        search_queries = list(dict.fromkeys(
            active_workflow.get("search_queries") or [plan.get("rewritten_query") or raw_q]
        ))
        st.markdown(f"**Search Queries Executed ({len(search_queries)}):**")
        for sq_text in search_queries:
            st.caption(f"• `{sq_text}`")

        st.markdown(f"**Retrieved Candidate Chunks ({len(all_chunks)}):**")
        if legacy_chunks and all_chunks:
            st.caption(
                "Showing saved answer-context chunks; full retrieval results were not "
                "stored for this earlier workflow."
            )
        if not all_chunks:
            st.caption("No candidate passages returned.")
        else:
            for idx, c in enumerate(all_chunks[:12], 1):
                cid = c.get("chunk_id", f"chk_{idx}")
                title = c.get("title") or "Document"
                page = c.get("page_number")
                page_str = f" · p.{page}" if page is not None else ""
                match_type = c.get("match_type", "vector")
                score = c.get("hybrid_score")
                score_str = f"score: {score:.3f}" if score is not None else f"dist: {c.get('distance', 0.0):.3f}"
                text_snippet = (c.get("text") or "").strip()[:180].replace("\n", " ")

                with st.expander(f"#{idx} [{match_type}] {title}{page_str} ({score_str})"):
                    st.caption(f"**Chunk ID:** `{cid}` | **Match:** `{match_type}` | **Relevance:** `{score_str}`")
                    st.text(c.get("text") or "")

    # -------------------------------------------------------------
    # STAGE 4: Top Chunks Selection (Curator)
    # -------------------------------------------------------------
    with st.expander(f"🏆 4. Top Chunks Selection ({len(curated_chunks)} context)", expanded=False):
        is_sufficient = active_workflow.get("sufficient") or active_workflow.get("evidence_sufficient", bool(curated_chunks))
        confidence = active_workflow.get("rerank_confidence", "standard")

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown(f"**Sufficiency:** {'✅ Sufficient' if is_sufficient else '⚠️ Low Evidence'}")
        with c_col2:
            st.markdown(f"**Rerank Confidence:** `{confidence}`")

        st.markdown(f"**Curated Evidence Context Blocks ({len(curated_chunks)}):**")
        for idx, block in enumerate(curated_chunks, 1):
            title = block.get("title") or "Doc"
            page = block.get("page_number")
            page_str = f" · Page {page}" if page is not None else ""
            st.markdown(f"**[{idx}]** 📄 `{title}{page_str}`")
            st.caption(f"{(block.get('text') or '').strip()[:150]}...")

    # -------------------------------------------------------------
    # STAGE 5: Answer Formation & Citations
    # -------------------------------------------------------------
    with st.expander("✍️ 5. Answer Formation & Citations", expanded=False):
        output_format = plan.get("output_format") or q_info.get("recommended_output_format") or "short_answer"
        st.markdown(f"**Output Format Target:** `{output_format}`")
        st.markdown(f"**Extracted Citation References:** `{citations or 'None'}`")

        if raw_answer and raw_answer != final_answer:
            st.markdown("**Initial Pre-Verification Draft:**")
            st.caption(raw_answer[:400] + ("..." if len(raw_answer) > 400 else ""))

        st.markdown("**Final Sanitized Output:**")
        st.caption(final_answer[:400] + ("..." if len(final_answer) > 400 else ""))

    # -------------------------------------------------------------
    # STAGE 6: Verification & Claims Audit
    # -------------------------------------------------------------
    with st.expander("🛡️ 6. Verification & Claims Audit", expanded=False):
        strictness = q_info.get("verification_strictness", "standard")
        st.markdown(f"**Verification Strictness:** `{strictness.upper()}`")

        claims = verdict.get("claims") or []
        if not claims:
            if verdict.get("skipped"):
                st.info("Verification skipped (non-factual query or conversational flow).")
            else:
                st.caption("No claim statements were extracted for audit.")
        else:
            supported_count = len([c for c in claims if c.get("supported") == "yes"])
            st.markdown(f"**Claims Verified:** `{supported_count}/{len(claims)} supported`")
            for idx, cl in enumerate(claims, 1):
                c_text = cl.get("claim") or cl.get("sentence") or ""
                sup = cl.get("supported") == "yes"
                icon = "✅" if sup else "❌"
                passages = cl.get("passage_ids") or []
                pass_str = f" (Citations: {passages})" if passages else ""
                st.markdown(f"{icon} **Claim {idx}:** {c_text}{pass_str}")

        missing = verdict.get("missing") or []
        if missing:
            st.warning("**Unsubstantiated Information Filtered:**")
            for m in missing:
                st.caption(f"• {m}")

    # -------------------------------------------------------------
    # STAGE 7: LLM Calls, Tokens & Timing Breakdown
    # -------------------------------------------------------------
    with st.expander("⏱️ 7. Timing & Token Telemetry", expanded=False):
        st.markdown("#### 🪙 Token Consumption")
        t_col1, t_col2, t_col3 = st.columns(3)
        with t_col1:
            st.metric("Total", _format_tokens(total_tokens))
        with t_col2:
            st.metric("Input", _format_tokens(prompt_tokens))
        with t_col3:
            st.metric("Output", _format_tokens(completion_tokens))

        call_details = token_usage.get("call_details") or []
        if call_details:
            st.markdown("**Model Calls Breakdown:**")
            for cd in call_details:
                c_model = cd.get("model", "default")
                c_stage = cd.get("agent") or "llm"
                c_tok = cd.get("total_tokens", 0)
                st.caption(f"• `{c_stage}` ({c_model}): **{c_tok}** tokens")

        st.markdown("---")
        st.markdown("#### ⏳ Stage Latency Breakdown")
        if not timings:
            st.caption("No granular timing blocks recorded.")
        else:
            sorted_timings = sorted(timings, key=lambda x: x.get("elapsed_seconds", 0), reverse=True)
            for t in sorted_timings:
                lbl = t.get("label", "stage")
                sec = t.get("elapsed_seconds", 0.0)
                cat = t.get("category", "")
                det = t.get("details", "")
                det_str = f" · {det}" if det else ""
                st.markdown(f"**{_format_seconds(sec)}** — `{cat}` {lbl}{det_str}")

    with st.expander("🛡️ 8. Output Rails", expanded=False):
        validation_status = "Passed" if output_validation.get("valid") else "Issues found"
        st.markdown(f"**Output structure:** {validation_status}")
        citation_check = output_validation.get("citation_validation") or {}
        citation_status = (
            "Complete" if citation_check.get("complete") else "Partial or unavailable"
        )
        st.caption(
            f"Citations checked: {citation_check.get('citations_checked', 0)} · "
            f"Provenance: {citation_status}"
        )
        for issue in output_validation.get("issues") or []:
            st.caption(f"• Issue: {issue}")
        for warning in output_validation.get("warnings") or []:
            st.caption(f"• Warning: {warning}")

        if summary_coverage:
            st.markdown("**Summary coverage**")
            st.caption(
                f"Sections: {summary_coverage.get('sections_covered', 0)}/"
                f"{summary_coverage.get('sections_checked', 0)} · "
                f"Important facts: {summary_coverage.get('facts_covered', 0)}/"
                f"{summary_coverage.get('facts_checked', 0)}"
            )
            if summary_coverage.get("facts_limit_reached"):
                st.caption("Fact coverage reached its configured check limit.")
            for section in summary_coverage.get("missing_sections") or []:
                st.caption(f"• Missing section: {section}")
            for fact in summary_coverage.get("missing_facts") or []:
                st.caption(f"• Missing fact: {fact.get('fact', '')}")

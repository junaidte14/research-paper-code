"""
AgentHarm Research Lab — single-page Streamlit app.
Fetch the dataset, pick a model, run the resumable paraphrase-stability
sweep, review/fix individual behaviors, and export results. Switch the
model at the top to re-run the same experiment against a different Groq
model — each model gets its own trajectory log, so results never mix.
"""

import json
import pandas as pd
import streamlit as st

import agentharm_lab as lab

st.set_page_config(page_title="AgentHarm Research Lab", layout="wide")
st.title("🧪 AgentHarm Research Lab")
st.caption("Resumable paraphrase-stability sweep against AgentHarm, switchable across Groq models.")

if not lab.GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart.")
    st.stop()

# ============================================================
# Model switcher
# ============================================================
try:
    available_models = lab.list_available_models()
except lab.GroqClientError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.subheader("⚙️ Model")
    model = st.selectbox("Active model (drives new runs + which log is shown)", options=available_models)
    max_steps = st.slider("Max steps per run", min_value=1, max_value=15, value=6)

    logged_models = lab.list_models_with_logs()
    if logged_models:
        st.caption("Models with existing data: " + ", ".join(logged_models))

# ============================================================
# 0. Data integrity
# ============================================================
st.subheader("🔧 Data integrity")
st.caption(f"Re-derives every logged decision for **{model}** from its saved transcript using the "
           "latest detection logic — no API calls, no cost.")
if st.button("🔄 Reclassify all logged decisions for this model"):
    result = lab.reclassify_log(model)
    st.success(f"Reclassified {result['changed']} of {result['total']} logged trajectories.")
    st.rerun()

# ============================================================
# 1. Dataset
# ============================================================
st.subheader("1️⃣ Dataset")
if not lab.HF_TOKEN:
    st.warning("HF_TOKEN not set — fetching will fail, but cached data can still be used.")

col1, col2, col3 = st.columns(3)
with col1:
    config_name = st.text_input("Config", value=lab.DEFAULT_CONFIG_NAME)
with col2:
    split = st.text_input("Split", value=lab.DEFAULT_SPLIT)
with col3:
    force_redownload = st.checkbox("Force re-download", value=False)

if st.button("⬇️ Fetch / verify locally"):
    try:
        entry = lab.fetch_dataset(config_name, split, force_redownload=force_redownload)
        st.success(f"Cached {entry['num_rows']} rows at `{entry['path']}`")
    except lab.DatasetError as e:
        st.error(str(e))

if not lab.is_cached(config_name, split):
    st.info("This config/split isn't cached yet — fetch it above before continuing.")
    st.stop()

rows = lab.load_local(config_name, split)
st.caption(f"{len(rows)} rows cached in {config_name}/{split}")

# ============================================================
# 2. Scope
# ============================================================
st.subheader("2️⃣ Scope")
scope_mode = st.radio("Which rows to work through", options=["Sampled worklist", "Every row in this split"], horizontal=True)
behavior_candidates = lab.build_behavior_index(model)

if scope_mode == "Sampled worklist":
    c1, c2 = st.columns(2)
    with c1:
        n_unstable = st.number_input("Unstable-by-format behaviors", min_value=0, max_value=44, value=10)
    with c2:
        n_stable = st.number_input("Stable control behaviors", min_value=0, max_value=44, value=10)
    sample = lab.select_sample(behavior_candidates, n_unstable=int(n_unstable), n_stable_controls=int(n_stable))
    scope_task_ids = {c.anchor_task_id for c in sample["combined"] if c.anchor_task_id}
else:
    scope_task_ids = None

scope_rows = []
for idx, row in enumerate(rows):
    task = lab.extract_task(row, fallback_id=f"{config_name}_{split}_{idx}")
    if scope_task_ids is None or task.task_id in scope_task_ids:
        scope_rows.append((idx, task))
st.caption(f"{len(scope_rows)} row(s) in scope for **{model}**.")

# ============================================================
# 3. Targets
# ============================================================
st.subheader("3️⃣ Targets")
t1, t2, t3, t4 = st.columns(4)
with t1:
    target_paraphrases = st.number_input("Paraphrases per behavior", min_value=1, max_value=10, value=3)
with t2:
    target_seeds = st.number_input("Seeds per variant", min_value=1, max_value=10, value=2)
with t3:
    gen_temperature = st.slider("Paraphrase temperature", min_value=0.0, max_value=1.5, value=0.8, step=0.1)
with t4:
    rows_per_batch = st.number_input("Rows to process this click", min_value=1, max_value=50, value=5)

st.caption("Paraphrases are shared across models (already-saved variants are reused, not regenerated) — "
           "only the seeded Track B runs are per-model, so switching models re-tests the same paraphrase set.")

# ============================================================
# 4. Progress
# ============================================================
st.subheader("4️⃣ Progress")
all_records = lab.load_all_trajectories(model)
baseline_by_id = {r["task_id"]: r for r in all_records if "variant_key" not in r.get("meta", {})}


def row_status(task):
    baseline = baseline_by_id.get(task.task_id)
    perturbations = lab.load_perturbations(task.task_id)
    result = lab.compute_behavior_result(model, task.task_id, all_records=all_records)
    needed_runs = (1 + len(perturbations)) * int(target_seeds)
    done_runs = sum(result.n_seeds_per_variant.values()) if result else 0

    if not baseline:
        stage = "baseline pending"
    elif len(perturbations) < target_paraphrases:
        stage = "paraphrases pending"
    elif done_runs < needed_runs:
        stage = "runs pending"
    else:
        stage = "complete"

    verdict = None
    if result:
        verdict = "unstable" if not result.cross_variant_stable else ("noisy" if result.noisy_variants else "stable")

    return {"task_id": task.task_id, "category": task.metadata.get("category"), "name": task.metadata.get("name"),
            "baseline": baseline["decision"] if baseline else None, "perturbations_saved": len(perturbations),
            "runs_done": done_runs, "runs_needed": needed_runs, "stage": stage, "verdict": verdict}


progress_rows = [row_status(task) for _, task in scope_rows]
st.dataframe(pd.DataFrame(progress_rows), use_container_width=True)

n_complete = sum(1 for r in progress_rows if r["stage"] == "complete")
c1, c2, c3 = st.columns(3)
c1.metric("Complete", f"{n_complete}/{len(progress_rows)}")
c2.metric("Unstable found", sum(1 for r in progress_rows if r["verdict"] == "unstable"))
c3.metric("Noisy (needs more seeds)", sum(1 for r in progress_rows if r["verdict"] == "noisy"))

usage_values = [r["usage"]["total_tokens"] for r in all_records if "usage" in r and r["usage"].get("total_tokens")]
avg_tokens = int(sum(usage_values) / len(usage_values)) if usage_values else 1500
remaining_rows = [r for r in progress_rows if r["stage"] != "complete"]
budget = lab.estimate_token_budget(len(remaining_rows), int(target_paraphrases), int(target_seeds), avg_tokens)
st.caption(f"Estimated remaining cost for {model}: ~{budget['estimated_total_tokens']:,} tokens across "
           f"{len(remaining_rows)} unfinished row(s) (avg {avg_tokens:,} tokens/run observed so far).")

# ============================================================
# 5. Process (resumable)
# ============================================================
st.subheader("5️⃣ Process")
st.caption("Does whatever's missing for the next unfinished rows. Safe to stop anytime; click again to continue.")

if st.button("▶️ Process next rows", type="primary"):
    task_lookup = {t.task_id: t for _, t in scope_rows}
    to_process_ids = [r["task_id"] for r in progress_rows if r["stage"] != "complete"][: int(rows_per_batch)]

    if not to_process_ids:
        st.success("Nothing left to process in this scope at current targets.")
    else:
        overall_progress = st.progress(0.0)
        status_line = st.empty()

        for i, task_id in enumerate(to_process_ids):
            task = task_lookup[task_id]
            status_line.info(f"Row {i + 1}/{len(to_process_ids)} — {task.task_id} ({task.metadata.get('name')})")
            tools_schema, tool_impl = lab.generate_stub_tools(task)

            if task.task_id not in baseline_by_id:
                try:
                    result = lab.run_agent(task.prompt, model, task_id=task.task_id, max_steps=max_steps,
                                            tools_schema=tools_schema, tool_impl=tool_impl)
                    lab.log_trajectory(model, result, extra_meta={
                        "config_name": config_name, "split": split, **task.metadata})
                    baseline_by_id[task.task_id] = result
                except Exception as e:
                    st.warning(f"{task.task_id}: baseline run failed — {e}")

            perturbations = lab.load_perturbations(task.task_id)
            if len(perturbations) < target_paraphrases:
                still_needed = int(target_paraphrases) - len(perturbations)
                try:
                    instructions = lab.paraphrase_instructions(task)
                    gen_result = lab.generate_paraphrases(task.prompt, model=model, n_variants=still_needed,
                                                            extra_instructions=instructions, temperature=gen_temperature)
                    if gen_result.valid_variants:
                        lab.save_perturbations(task.task_id, gen_result.valid_variants, generation_model=model,
                                                metadata={"config_name": config_name, "split": split})
                    if gen_result.refused_variants:
                        lab.record_generation_refusal(task.task_id, len(gen_result.refused_variants),
                                                        len(gen_result.valid_variants) + len(gen_result.refused_variants))
                    perturbations = lab.load_perturbations(task.task_id)
                except Exception as e:
                    st.warning(f"{task.task_id}: paraphrase generation failed — {e}")

            already_logged = {r.get("task_id") for r in lab.load_all_trajectories(model)}
            variants = [("original", "original", task.task_id, task.prompt)] + [
                (p.perturbation_type, p.perturbation_id, f"{task.task_id}__{p.perturbation_id}", p.prompt)
                for p in perturbations
            ]
            for variant_type, variant_key, base_id, prompt_text in variants:
                for seed_idx in range(1, int(target_seeds) + 1):
                    run_task_id = f"{base_id}__seed{seed_idx}"
                    if run_task_id in already_logged:
                        continue
                    try:
                        result = lab.run_agent(prompt_text, model, task_id=run_task_id, max_steps=max_steps,
                                                tools_schema=tools_schema, tool_impl=tool_impl)
                        lab.log_trajectory(model, result, extra_meta={
                            "config_name": config_name, "split": split, "original_task_id": task.task_id,
                            "variant_type": variant_type, "variant_key": variant_key, "seed": seed_idx, **task.metadata})
                    except Exception as e:
                        st.warning(f"{task.task_id} / {variant_key} seed {seed_idx} failed — {e}")

            overall_progress.progress((i + 1) / len(to_process_ids))

        status_line.empty()
        st.success(f"Processed {len(to_process_ids)} row(s) for {model}.")
        st.rerun()

# ============================================================
# 6. Review / fix a specific row
# ============================================================
st.subheader("6️⃣ Review or fix a specific row")
task_lookup = {t.task_id: t for _, t in scope_rows}
task_ids_in_scope = list(task_lookup.keys())

if task_ids_in_scope:
    sel_task_id = st.selectbox("Task", options=task_ids_in_scope)
    sel_task = task_lookup[sel_task_id]
    st.text_area("Prompt", value=sel_task.prompt, height=80, disabled=True)

    perturbations = lab.load_perturbations(sel_task_id)
    for p in perturbations:
        with st.container(border=True):
            st.markdown(f"**`{p.perturbation_id}`**")
            st.write(p.prompt)
            if st.button("🗑️ Delete this variant", key=f"del_{p.perturbation_id}"):
                lab.delete_perturbation(sel_task_id, p.perturbation_id)
                st.rerun()

    extra_seeds = st.number_input(f"Bump seeds for {sel_task_id} to:", min_value=1, max_value=15,
                                   value=int(target_seeds), key=f"bump_{sel_task_id}")
    if st.button(f"Run additional seeds for {sel_task_id} on {model}"):
        tools_schema, tool_impl = lab.generate_stub_tools(sel_task)
        variants = [("original", "original", sel_task_id, sel_task.prompt)] + [
            (p.perturbation_type, p.perturbation_id, f"{sel_task_id}__{p.perturbation_id}", p.prompt)
            for p in perturbations
        ]
        already_logged = {r.get("task_id") for r in lab.load_all_trajectories(model)}
        pending = [
            (vt, vk, f"{base_id}__seed{s}", prompt_text, s)
            for vt, vk, base_id, prompt_text in variants
            for s in range(1, int(extra_seeds) + 1)
            if f"{base_id}__seed{s}" not in already_logged
        ]
        prog = st.progress(0.0)
        for i, (vt, vk, run_task_id, prompt_text, seed_idx) in enumerate(pending):
            try:
                result = lab.run_agent(prompt_text, model, task_id=run_task_id, max_steps=max_steps,
                                        tools_schema=tools_schema, tool_impl=tool_impl)
                lab.log_trajectory(model, result, extra_meta={
                    "config_name": config_name, "split": split, "original_task_id": sel_task_id,
                    "variant_type": vt, "variant_key": vk, "seed": seed_idx, **sel_task.metadata})
            except Exception as e:
                st.warning(f"seed run failed — {e}")
            prog.progress((i + 1) / max(len(pending), 1))
        st.success(f"Ran {len(pending)} additional seed(s) on {model}.")
        st.rerun()

    result = lab.compute_behavior_result(model, sel_task_id)
    if result:
        st.json({"majority_decisions": result.majority_decisions, "noisy_variants": result.noisy_variants,
                  "cross_variant_stable": result.cross_variant_stable,
                  "safety_relevant_instability": result.safety_relevant_instability})
        related = [r for r in lab.load_all_trajectories(model) if r.get("meta", {}).get("original_task_id") == sel_task_id]
        related = lab.dedupe_latest_per_seed(related)
        if related:
            st.download_button(f"⬇️ Download full transcripts for {sel_task_id} on {model} (JSON)",
                                data=json.dumps(related, indent=2),
                                file_name=f"agentharm_{lab.model_slug(model)}_{sel_task_id}_transcripts.json",
                                mime="application/json")

# ============================================================
# 7. Export
# ============================================================
st.subheader("7️⃣ Export results")
all_results = lab.aggregate_results(model)
export_rows = [{
    "behavior": r.original_task_id, "category": r.category, "name": r.name,
    "variants_run": r.n_variants, "seed_noise": "Yes" if r.noisy_variants else "No",
    "cross_variant_stable": "Stable" if r.cross_variant_stable else "Unstable",
    "safety_relevant": "Yes" if r.safety_relevant_instability else "No",
    "original_decision": r.overall_decision, "decisions_by_variant": str(r.majority_decisions),
    "noisy_variants": ", ".join(r.noisy_variants) if r.noisy_variants else "",
} for r in all_results]
df_export = pd.DataFrame(export_rows)
st.dataframe(df_export, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.download_button(f"⬇️ Results summary for {model} (CSV)", data=df_export.to_csv(index=False),
                        file_name=f"agentharm_{lab.model_slug(model)}_results.csv", mime="text/csv")
with c2:
    full_bundle = {
        "model": model, "results_summary": export_rows,
        "all_seeded_trajectories": lab.dedupe_latest_per_seed(
            [r for r in lab.load_all_trajectories(model) if "variant_key" in r.get("meta", {})]),
    }
    st.download_button(f"⬇️ Full bundle for {model} (JSON)", data=json.dumps(full_bundle, indent=2),
                        file_name=f"agentharm_{lab.model_slug(model)}_full_bundle.json", mime="application/json")
import os
import sys
import json
import asyncio
import httpx
from typing import List, Dict, Any

# Simple INR conversion and cost assumptions (can be tuned via env)
USD_TO_INR = float(os.getenv("USD_TO_INR", "84.0"))
# Fall back rates matching tester defaults
ASR_RATE_PER_MIN = {
    "google": float(os.getenv("ASR_RATE_GOOGLE_PER_MIN", "0.018")),
    "elevenlabs": float(os.getenv("ASR_RATE_ELEVEN_PER_MIN", "0.030")),
    "sarvam": float(os.getenv("ASR_RATE_SARVAM_PER_MIN", "0.010")),
}
TTS_RATE_PER_CHAR = {
    "google": float(os.getenv("TTS_RATE_GOOGLE_PER_CHAR", "0.000016")),
    "elevenlabs": float(os.getenv("TTS_RATE_ELEVEN_PER_CHAR", "0.00003")),
}

BOT_URL = os.getenv("BOT_URL", "http://127.0.0.1:8000")


async def run_persona_tests() -> Dict[str, Any]:
    """Run all personas using the clean tester and build a summary-like dict.
    Returns:
        dict: summary with entity_f1, slot_completion, ranking-like info
    """
    # Ensure server is up; if not, start it temporarily
    async def is_up() -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{BOT_URL}/api/v1/health")
                return r.status_code == 200
        except Exception:
            return False

    proc = None
    if not await is_up():
        import subprocess, time
        host = BOT_URL.split("://", 1)[-1].split(":")[0]
        port = BOT_URL.rsplit(":", 1)[-1]
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            host,
            "--port",
            port,
        ]
        proc = subprocess.Popen(cmd)
        time.sleep(1.5)

    # Use the clean tester
    sys.path.append("test_harness")
    from http_conversation_tester_new import HTTPPersonaConversationTester  # type: ignore

    tester = HTTPPersonaConversationTester(bot_base_url=BOT_URL)

    # Run all personas serially
    results = []
    for persona_key in tester.personas.keys():
        res = await tester.run_persona_test(persona_key)
        results.append(res)

    # Build a simple summary similar to old evaluator expectations
    # We will assemble per-slot F1 etc from res.entity_extraction_metrics
    # and create a ranking-like macro from res.job_matching_metrics
    per_slot_agg: Dict[str, Dict[str, float]] = {}
    per_persona_completion = []
    p_at3_list, r_at3_list = [], []

    # Estimate costs
    costs_usd = []

    for res in results:
        ent = res.entity_extraction_metrics or {"per_slot": {}, "macro_f1": 0.0, "micro_f1": 0.0, "total_tp": 0, "total_fp": 0, "total_fn": 0}
        # Aggregate per-slot TP/FP/FN
        for slot, m in (ent.get("per_slot", {}) or {}).items():
            d = per_slot_agg.setdefault(slot, {"tp": 0, "fp": 0, "fn": 0})
            d["tp"] += m.get("tp", 0)
            d["fp"] += m.get("fp", 0)
            d["fn"] += m.get("fn", 0)

        # Completion proxy: if conversation completed, count as full; else use fraction of non-empty fields
        profile_slots = ["pincode", "availability_date", "preferred_shift", "expected_salary", "languages", "has_two_wheeler", "total_experience_months", "confirmation"]
        filled = set()
        # Use last conversation turn profile snapshot if available
        last = res.conversation_turns[-1] if getattr(res, "conversation_turns", []) else None
        prof = (last.candidate_profile if last else {}) or {}
        for slot in profile_slots:
            if slot == "confirmation":
                if prof.get("conversation_completed"):
                    filled.add("confirmation")
            else:
                if prof.get(slot) not in (None, [], ""):
                    filled.add(slot)
        completion_rate = len(filled.intersection(set(profile_slots))) / len(profile_slots)
        per_persona_completion.append({
            "persona": tester.personas.get(res.persona_key, {}).get("name", res.persona_key),
            "completion_rate": completion_rate,
            "turns_used": len(getattr(res, "conversation_turns", [])) or 0,
        })

        # Eligibility proxy: treat job_matching_metrics f1/precision/recall as overall; derive P@3/R@3 if present
        jm = res.job_matching_metrics or {}
        if "precision" in jm:
            p_at3_list.append(jm.get("precision", 0.0))
            r_at3_list.append(jm.get("recall", 0.0))

        # Rough cost estimate by text lengths and time
        turns = getattr(res, "conversation_turns", []) or []
        est_sec = sum(max(0.1, (t.latency_ms or 0) / 1000.0) for t in turns)
        asr_usd = (ASR_RATE_PER_MIN["google"] / 60.0) * est_sec
        tts_chars = sum(len((t.bot_response or "")) for t in turns)
        tts_provider = "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "google"
        tts_usd = TTS_RATE_PER_CHAR.get(tts_provider, 0.0) * tts_chars
        costs_usd.append(asr_usd + tts_usd)

    # Compute F1 per slot from aggregated counts
    def compute_f1(tp: int, fp: int, fn: int) -> float:
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 0.0 if (precision + recall == 0) else (2 * precision * recall / (precision + recall))

    per_slot_out: Dict[str, Dict[str, float]] = {}
    f1_vals = []
    for slot, counts in per_slot_agg.items():
        f1 = compute_f1(counts["tp"], counts["fp"], counts["fn"])
        per_slot_out[slot] = {**counts, "f1": f1}
        f1_vals.append(f1)

    summary: Dict[str, Any] = {
        "entity_f1": {
            "per_slot": per_slot_out,
            "macro_f1": (sum(f1_vals) / len(f1_vals)) if f1_vals else 0.0,
        },
        "slot_completion": {
            "per_persona": per_persona_completion,
            "overall_completion_rate": (sum(r["completion_rate"] for r in per_persona_completion) / len(per_persona_completion)) if per_persona_completion else 0.0,
        },
        "ranking": {
            "overall": {
                "macro_precision_at_3": (sum(p_at3_list) / len(p_at3_list)) if p_at3_list else 0.0,
                "macro_recall_at_3": (sum(r_at3_list) / len(r_at3_list)) if r_at3_list else 0.0,
            }
        }
    }

    # Attach INR estimate
    summary["costs"] = {
        "usd_to_inr": USD_TO_INR,
        "avg_usd_per_candidate": (sum(costs_usd) / len(costs_usd)) if costs_usd else 0.0,
        "avg_inr_per_candidate": ((sum(costs_usd) / len(costs_usd)) * USD_TO_INR) if costs_usd else 0.0,
    }

    # Pull server-wide p50/p95 latency
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{BOT_URL}/api/v1/metrics/dashboard")
            if resp.status_code == 200:
                dash = resp.json()
                summary["latency"] = {
                    "p50_ms": float(dash.get("p50_latency_ms", 0.0)),
                    "p95_ms": float(dash.get("p95_latency_ms", 0.0)),
                }
    except Exception:
        pass

    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    ent = summary.get("entity_f1", {})
    per = ent.get("per_slot", {})
    macro_f1 = ent.get("macro_f1", 0.0)
    slot = summary.get("slot_completion", {})
    overall_comp = slot.get("overall_completion_rate", 0.0)
    ranking = summary.get("ranking", {}).get("overall", {})
    p50 = summary.get("latency", {}).get("p50_ms", 0.0)
    p95 = summary.get("latency", {}).get("p95_ms", 0.0)
    costs = summary.get("costs", {})

    print("\n=== Eval Metrics ===")
    print("Entity F1 (per slot):")
    for slot_name, stats in per.items():
        print(f" - {slot_name}: F1 {stats.get('f1', 0.0):.3f} (TP {stats.get('tp',0)}, FP {stats.get('fp',0)}, FN {stats.get('fn',0)})")
    print(f"Macro-F1: {macro_f1:.3f}")
    print("\nSlot completion (<=10 turns):")
    print(f"Overall completion rate: {overall_comp*100:.1f}%")
    for row in slot.get("per_persona", []):
        print(f" - {row['persona']}: {row['completion_rate']*100:.1f}% (turns_used={row.get('turns_used', 0)})")
    print("\nEligibility (macro):")
    print(f"P@3 {ranking.get('macro_precision_at_3',0.0):.3f} | R@3 {ranking.get('macro_recall_at_3',0.0):.3f}")
    print(f"Latency: p50 {p50:.0f} ms | p95 {p95:.0f} ms")
    print("Estimated cost:")
    print(f"Avg ₹/candidate: ₹{costs.get('avg_inr_per_candidate', 0.0):.2f} (USD {costs.get('avg_usd_per_candidate', 0.0):.4f} @ {costs.get('usd_to_inr', 0)} INR/USD)")


def main() -> None:
    summary = asyncio.run(run_persona_tests())
    print_summary(summary)


if __name__ == "__main__":
    main() 
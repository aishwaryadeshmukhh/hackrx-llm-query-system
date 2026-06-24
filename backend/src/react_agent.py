"""
react_agent.py — ReAct (Reason + Act) agent loop for the Insurance RAG pipeline.

Each query runs through a max of MAX_STEPS iterations:
  1. LLM receives the question + all previous steps (thoughts + retrieved evidence)
  2. LLM outputs either:
       Thought + Action  →  tool is called, observation added, loop continues
       Final Answer      →  loop exits, answer is returned

The reasoning_trace (list of step dicts) is returned alongside the final answer
so callers can include it in the API response for visibility in Swagger / Streamlit.

Tool routing
------------
The agent can call any of the 5 tools in agent_tools.py:
  search_policy, lookup_exclusions, check_waiting_period, get_definitions, compare_policies

A simple query (clear coverage question with no ambiguity) exits in 1-2 steps.
A complex query (exclusion + waiting period both relevant) may use 3-5 steps.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_tools import ToolExecutor

MAX_STEPS = 6

# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an insurance policy analyst for an Indian health insurance policy (Bajaj Allianz Global Health Care). Answer coverage questions using the retrieval tools below.

Available tools and their EXACT argument names:
- search_policy(query, policy_name=null, top_k=15): General semantic search
  Args example: {"query": "cataract surgery coverage"}
- lookup_exclusions(procedure_or_condition): Search exclusion and conditional-coverage clauses
  Args example: {"procedure_or_condition": "bariatric surgery"}
- check_waiting_period(benefit_type): Search waiting period clauses
  Args example: {"benefit_type": "pre-existing disease"}
- get_definitions(term): Fetch glossary/definition entries
  Args example: {"term": "day care procedure"}

STRICT OUTPUT FORMAT — output EXACTLY ONE of these two blocks per step, nothing else:

When you need more information:
Thought: <one sentence on what you need next>
Action: <tool_name>
Args: <JSON object with the EXACT argument name shown above>

When you have enough evidence:
Thought: <one sentence summarising your conclusion>
Final Answer: {"decision": "covered|not_covered|partial|unclear", "confidence": 0.0, "answer": "3-5 sentences citing the specific clause, any conditions, limits, and waiting periods that apply", "justification": "exact section name and key clause text from the observation", "relevant_clauses": [{"section": "...", "content": "direct quote from the policy clause", "page": null}]}

CRITICAL RULES:
- 2-3 tool calls is the target; 6 is the absolute max
- After each tool call, check: does the observation fully answer the question including all applicable limits, conditions, and waiting periods? If yes → Final Answer. If the question mentions specific patient details (BMI, comorbidity, policy duration, plan type) that the first observation does not address → make one more targeted search.
- When you have found all relevant clauses, STOP searching and give Final Answer — do not repeat searches or search for confirmation of what you already know.
- NEVER conclude not_covered from a general exclusion clause alone if the question gives specific patient details — the policy may have conditional coverage clauses that override the exclusion for those exact details. Always search for the specific condition before concluding.
- decision must be exactly: covered, not_covered, partial, or unclear
- In your answer field, quote actual limits/durations/conditions from the retrieved text — not vague summaries
- In relevant_clauses, copy the actual clause text — do not paraphrase. NEVER write content that is not a direct quote from the observation text.
- Do NOT output any text outside the two formats above
- Do NOT repeat a tool call you already made
- Use policy-domain terminology: "area of cover" not "outside usa", "sum insured" not "coverage limit", "hospitalisation" not "hospital stay", "pre-existing disease" not "prior condition"
- This is a HEALTH insurance policy — it covers medical treatment expenses only. It does NOT provide: accidental death lump sum, life cover, personal accident benefit, disability income, critical illness lump sum. If the question asks about one of these and no specific benefit clause is found in the observations, decision = not_covered, confidence = 0.90. Do not infer coverage from unrelated clauses.
- If after 2 tool calls you have not found a clause that directly addresses the question, conclude with the best available decision — do not keep searching for something that may not exist in the policy.

═══════════════════════════════════════════════
DECISION LOGIC
═══════════════════════════════════════════════

WAITING PERIOD RULES:
- 30-day general wait: applies to all illness claims in first 30 days. Waived if insured had continuous coverage >12 months. Accidents always exempt.
- 24-month specified disease wait (Code-Excl02): applies to 35 listed conditions including cataract, hernia, bariatric surgery, joint replacement, all listed conditions. Accidents exempt. If a condition also qualifies as PED, the LONGER of 24-month or 36-month applies.
- 36-month PED wait (Code-Excl01): applies to pre-existing diseases (diagnosed or treated within 48 months before first policy date). Portability credit reduces this.
- Waiting period re-starts on enhanced Sum Insured for the enhanced portion only.
- At exactly N months elapsed (e.g. "3 years" = 36 months), the waiting period HAS elapsed → covered.

EXCLUSION-WITH-EXCEPTION RULES (always check for exceptions before deciding not_covered):
- Bariatric/obesity surgery (Code-Excl06): Default excluded. COVERED if ALL met: doctor-advised, clinically supported, age ≥18, AND (BMI ≥40 OR BMI ≥35 with: obesity cardiomyopathy, coronary heart disease, severe sleep apnoea, or uncontrolled Type 2 Diabetes). Note: bariatric is also in the 24-month Excl02 list — BOTH the BMI criteria AND the 24-month wait must be satisfied.
- Cosmetic/plastic surgery (Code-Excl08): Default excluded. COVERED if reconstruction after Accident, Burns, or Cancer; OR medically necessary to remove direct health risk (certified by treating doctor).
- Dental treatment: Default excluded. COVERED as inpatient emergency only if due to Accident requiring hospitalisation, treatment starts within 24 hours. Follow-up dental, implants, orthodontics remain excluded even in accident context. International dental plan add-on: covered with mandatory 20% co-payment; implants and orthodontics still permanently excluded.
- Congenital anomalies: External = permanent exclusion. Internal congenital disease = 24-month wait (eventually coverable). Haematopoietic stem cells for bone marrow transplant = carved out, covered.
- War/terrorism: War, civil war, hostilities = permanent exclusion. Act of Terrorism (with police charge sheet filed) = COVERED.
- Vaccination: Default excluded. COVERED if post-bite treatment OR prescribed by doctor as part of hospitalisation/day care treatment.
- Circumcision: Default excluded. COVERED if required to treat Illness or Accidental injury.
- Dietary supplements: Default excluded. COVERED if prescribed by doctor as part of hospitalisation or day care treatment.
- Sleep disorders: Insomnia, narcolepsy, snoring, bruxism = excluded. Inpatient treatment for obstructive sleep apnoea = COVERED. CPAP machine for home use = permanently excluded.
- Hair loss: Default excluded. Hair loss due to cancer treatment = COVERED.
- Tumour marker tests: Default excluded. COVERED when medically necessary during cancer investigation/treatment.
- Refractive error: <7.5 dioptres = permanent exclusion. ≥7.5 dioptres recommended by Ophthalmologist for medical reasons = covered after 24-month wait.
- Tumours/cysts/nodules/polyps: Benign = 24-month wait. Malignant tumours = NOT in the waiting period list, covered without waiting period.
- Maternity/childbirth (Code-Excl18): Excluded. Ectopic pregnancy = COVERED (explicit carve-out). Miscarriage due to Accident = COVERED. Miscarriage otherwise = excluded. IVF/ART/surrogacy = permanently excluded.
- Mental illness: Inpatient treatment COVERED for ICD-10 codes F00-F09, F20-F99 (except F10-F19). F10-F19 (alcohol, substance abuse, addiction) = PERMANENTLY EXCLUDED. Must be in recognised psychiatric unit, diagnosed by psychiatrist/psychologist. OPD mental illness = excluded. Autism admissions in specialised educational facilities = excluded.
- Hazardous sports (Code-Excl09): Only excluded for PROFESSIONAL participation. Amateur/recreational = covered.
- Intentional travel for treatment: If the insured specifically travelled outside India to seek treatment for a known condition = NOT COVERED under international cover.
- Excluded hospital emergency carve-out (Code-Excl11): Normally excluded providers. Life-threatening emergency or Accident = expenses up to stabilisation are PARTIAL/COVERED, not the full claim.

PARTIAL COVERAGE RULES:
- decision = partial when: coverage exists but with a sub-limit cap, co-payment, or deductible that reduces the payout
- Dental plan: mandatory 20% co-payment on every claim, cannot be waived
- International pre-approval missed: if treatment proven medically necessary, only 80% payable (20% penalty)
- Emergency treatment outside area of cover (Section 12, Imperial Plus + Excluding USA only): covered up to 6 weeks per trip within Sum Insured, treatment must start within 24 hours of Emergency. Maternity/childbirth permanently excluded even here. Decision = partial (6-week cap is a sub-limit).
- Room rent (International): capped at single private air-conditioned room; deluxe/suite upgrade causes proportionate disallowance on associated charges.
- Rehabilitation: sub-limited (INR 50,000 domestic; USD 750 Imperial international; USD 2,300 Imperial Plus international). Must start within 14 days of acute discharge.

PLAN TIER RULES (Imperial vs Imperial Plus):
- OPD benefits (out-patient, physiotherapy, alternate treatment) = Imperial Plus ONLY
- Emergency treatment outside area of cover (Section 12) = Imperial Plus ONLY, and only if "Excluding USA" cover is opted
- Medical evacuation, repatriation of mortal remains, inpatient cash benefit, palliative care, parent accommodation with hospitalised child = Imperial Plus ONLY
- Air ambulance: Imperial = USD 7,500 cashless only. Imperial Plus = up to full inpatient SI with evacuation.
- If a question mentions a benefit that is Imperial Plus only and the insured has Imperial plan → not_covered for that benefit.

GEOGRAPHIC CONTEXT:
- Base country = India. "Area of cover" = geographic zone of the plan.
- "Excluding USA" plan = worldwide except USA/Canada. Insured IS covered internationally (non-USA) but NOT in USA for routine/planned treatment.
- Section 12 "Emergency treatment outside area of cover" (Imperial Plus + Excluding USA only):
  * The BENEFIT clause says "We will pay… treatment of medical emergencies outside Your area of cover… up to six weeks per trip". This grants cover.
  * The EXCLUSION within Section 12 says "Cover is not provided for curative or follow-up non-Emergency treatment". This only excludes follow-up/curative — not the initial emergency.
  * These are TWO separate parts of the same clause. If you retrieve the exclusion text, do NOT apply it as the top-level decision — it is a sub-exclusion within an otherwise-covered benefit.
  * For a genuine medical emergency in USA on an Excluding USA plan: decision = partial (covered up to 6 weeks, within Sum Insured, treatment must start within 24 hours).
  * For curative/follow-up treatment in USA that is not an emergency: decision = not_covered (sub-exclusion applies).
  * For maternity/childbirth in USA: decision = not_covered (explicitly excluded even under Section 12).
  * Always search for Section 12 benefit text ("emergency treatment outside area of cover benefit We will pay") to get the positive grant clause, not just the exclusion chunk.

CONFIDENCE:
- Confidence reflects certainty of the decision, NOT whether coverage is generous.
- Found exact clause and applied it correctly → 0.80–0.95
- decision = unclear (policy genuinely doesn't address scenario) → confidence < 0.60
- Borderline waiting period (e.g. "exactly 36 months") → 0.70–0.80

PRE/POST HOSPITALISATION:
- Domestic: 60 days pre, 180 days post
- International: 45 days pre, 90 days post
- Must be for the same condition as the inpatient claim

MORATORIUM — 8 years:
- After 8 continuous years, no look-back applied (no contestability for undisclosed PED)
- Sub-limits, co-payments, room rent limits still apply after moratorium
"""

_STEP_PROMPT_TEMPLATE = """Question: {question}

{steps_so_far}
{current_observation}
Next step (output ONLY Thought+Action+Args OR Thought+Final Answer):"""

_OBSERVATION_TEMPLATE = """Step {step_num}:
Thought: {thought}
Action: {action}({args_json})
Observation: {observation}
"""

_FINAL_ANSWER_SCHEMA = {
    "decision": "unclear",
    "confidence": 0.0,
    "answer": "",
    "justification": "",
    "relevant_clauses": [],
}


# ── Parser ───────────────────────────────────────────────────────────────────

def _parse_llm_step(text: str) -> Dict[str, Any]:
    """
    Parse one LLM output step into a structured dict.

    Returns one of:
      {"type": "action",  "thought": str, "tool": str, "args": dict}
      {"type": "answer",  "thought": str, "answer": dict}
      {"type": "parse_error", "raw": str}
    """
    text = text.strip()

    # Extract Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    # Check for Final Answer
    fa_match = re.search(r"Final Answer:\s*(\{.*\})", text, re.DOTALL)
    if fa_match:
        try:
            answer_json = json.loads(fa_match.group(1))
        except json.JSONDecodeError:
            # Try to extract JSON more aggressively
            raw = fa_match.group(1)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            try:
                answer_json = json.loads(raw[start:end])
            except json.JSONDecodeError:
                answer_json = dict(_FINAL_ANSWER_SCHEMA)
                answer_json["answer"] = raw
        # Ensure all required fields
        for k, v in _FINAL_ANSWER_SCHEMA.items():
            answer_json.setdefault(k, v)
        return {"type": "answer", "thought": thought, "answer": answer_json}

    # Check for Action — handle both:
    #   Action: tool_name\nArgs: {...}       (preferred)
    #   Action: tool_name({"key": "val"})    (Llama sometimes outputs this)
    action_match = re.search(r"Action:\s*(\w+)", text)
    if action_match:
        tool = action_match.group(1).strip()
        args = {}

        # Format 1: separate Args: line
        args_match = re.search(r"Args:\s*(\{.*\})", text, re.DOTALL)
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                pass

        # Format 2: inline Action: tool_name({...})
        if not args:
            inline_match = re.search(r"Action:\s*\w+\s*\((\{.*?\})\)", text, re.DOTALL)
            if inline_match:
                try:
                    args = json.loads(inline_match.group(1))
                except json.JSONDecodeError:
                    pass

        # Format 3: key="value" or key: value anywhere after Action line
        if not args:
            kv_match = re.search(r'(?:query|procedure_or_condition|benefit_type|term)["\s:=]+(["\']?)([^"\'\n,}]+)\1', text)
            if kv_match:
                # Map whichever key was found to the right param name
                for param in ["procedure_or_condition", "benefit_type", "term", "query"]:
                    pm = re.search(rf'{param}["\s:=]+(["\']?)([^"\'\n,}}]+)\1', text)
                    if pm:
                        args[param] = pm.group(2).strip()

        return {"type": "action", "thought": thought, "tool": tool, "args": args}

    return {"type": "parse_error", "raw": text}


def _format_chunks_for_observation(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks for the LLM observation.

    Top 2 chunks: full text up to 1200 chars (the ones most likely to contain the answer).
    Chunks 3-6: section header + first 120 chars only (context breadcrumbs, not full text).
    This keeps the observation readable while cutting token usage by ~65% vs 8 full chunks.
    """
    if not chunks:
        return "No relevant chunks found."
    lines = []
    for i, c in enumerate(chunks[:6], 1):
        doc     = c.get("document", c.get("document_name", "unknown"))
        page    = c.get("page", c.get("page_number", "?"))
        score   = c.get("score", 0.0)
        section = c.get("section", "")
        text    = c.get("text", c.get("content", ""))
        header  = f" | Section: {section}" if section else ""

        if i <= 2:
            # Full text for top 2 results
            lines.append(f"[{i}] {doc} p.{page} (score={score:.3f}){header}\n    {text[:1200]}")
        else:
            # Header + snippet for remaining results
            lines.append(f"[{i}] {doc} p.{page} (score={score:.3f}){header} — {text[:120]}…")
    return "\n\n".join(lines)


def _compress_observation(tool_name: str, args: dict, observation: str) -> str:
    """
    Produce a compact 1-2 line summary of a completed step for the steps_so_far history.

    The full observation is shown to the LLM at the time of the step. For subsequent
    steps, only a compressed version is kept in context so the history doesn't balloon.
    """
    # Extract the highest-scoring section names and first clause sentence from observation
    sections = re.findall(r"Section:\s*([^\n|]+)", observation)
    section_str = "; ".join(s.strip() for s in sections[:3]) if sections else ""

    # Pull first sentence of the top chunk text (after the header line)
    first_text_match = re.search(r"\n\s+(.+?)(?:\n|$)", observation)
    first_sentence = ""
    if first_text_match:
        raw = first_text_match.group(1).strip()
        # Take up to first period or 200 chars
        dot = raw.find(".")
        first_sentence = raw[:dot + 1] if 0 < dot < 200 else raw[:200]

    arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    summary = f"{tool_name}({arg_str})"
    if section_str:
        summary += f" → sections: {section_str}"
    if first_sentence:
        summary += f" | key clause: {first_sentence}"
    return summary


# ── Decision correction ──────────────────────────────────────────────────────

_PARTIAL_INDICATORS = [
    "up to six weeks", "up to 6 weeks", "six weeks per trip", "6 weeks per trip",
    "sub-limit", "sublimit", "co-payment", "copayment", "20% co-payment",
    "proportionate deduction", "proportionate disallowance",
    "80% payable", "only 80%",
    "subject to a limit", "subject to sub-limit",
    "capped at", "maximum benefit amount",
]

def _correct_decision(answer: dict) -> dict:
    """
    Upgrade covered → partial when the answer text contains sub-limit language.
    The LLM often picks 'covered' when it should be 'partial' because it focuses
    on whether the claim is payable rather than whether a cap reduces the payout.
    """
    if answer.get("decision") != "covered":
        return answer
    text = (answer.get("answer", "") + " " + answer.get("justification", "")).lower()
    if any(indicator in text for indicator in _PARTIAL_INDICATORS):
        answer = dict(answer)
        answer["decision"] = "partial"
        print("[react] Decision corrected: covered → partial (sub-limit language detected)")
    return answer


# ── ReAct loop ───────────────────────────────────────────────────────────────

def run_react_loop(
    question: str,
    llm,
    tool_executor: "ToolExecutor",
    max_steps: int = MAX_STEPS,
    gemini_model=None,
    on_step: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run the ReAct loop for one question.

    Returns:
        {
            "answer": {decision, confidence, answer, justification, relevant_clauses},
            "reasoning_trace": [
                {
                    "step": int,
                    "thought": str,
                    "action": str | "final_answer",
                    "args": dict | {},
                    "observation": str | null   # null for final_answer step
                }
            ],
            "steps_taken": int,
            "status": "success" | "max_steps_reached" | "error"
        }
    """
    steps_so_far = ""       # compressed history of all completed steps
    current_observation = "" # full observation from the most recent tool call
    reasoning_trace = []
    # Track previous tool calls as (tool_name, canonical_args_json) to reliably detect repeats
    _prev_calls: set = set()

    system = _SYSTEM_PROMPT.replace("{max_steps}", str(max_steps))

    for step_num in range(1, max_steps + 1):
        prompt = f"{system}\n\n{_STEP_PROMPT_TEMPLATE.format(question=question, steps_so_far=steps_so_far, current_observation=current_observation)}"

        # Call LLM (Groq-first, Gemini fallback)
        try:
            from .llm_client import call_llm
            raw_text = call_llm(
                llm, gemini_model,
                [{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2,
            ) or ""
        except Exception as e:
            print(f"❌ ReAct LLM call failed at step {step_num}: {e}")
            return _error_result(question, str(e), reasoning_trace)

        if not raw_text:
            print(f"⚠️ Empty LLM response at step {step_num}")
            break

        parsed = _parse_llm_step(raw_text)
        print(f"🤔 Step {step_num} [{parsed['type']}]: {parsed.get('thought', '')[:80]}")

        if parsed["type"] == "answer":
            trace_entry = {
                "step": step_num,
                "thought": parsed["thought"],
                "action": "final_answer",
                "args": {},
                "observation": None,
            }
            reasoning_trace.append(trace_entry)
            corrected = _correct_decision(parsed["answer"])
            if on_step:
                on_step({"type": "answer", **corrected, "trace_entry": trace_entry})
            return {
                "answer": corrected,
                "reasoning_trace": reasoning_trace,
                "steps_taken": step_num,
                "status": "success",
            }

        if parsed["type"] == "action":
            tool_name = parsed["tool"]
            args = parsed["args"]

            # Reliable repeat detection using an in-memory set (not regex on text buffer)
            call_key = (tool_name, json.dumps(args, sort_keys=True))
            if call_key in _prev_calls:
                steps_remaining = max_steps - step_num
                if steps_remaining >= 2:
                    # Still have budget — redirect to a complementary tool instead of cutting off
                    print(f"[react] Repeat {tool_name} at step {step_num} — redirecting to search_policy")
                    # Build a contextual redirect — suggest a query relevant to the args
                    prev_query = args.get("query") or args.get("procedure_or_condition") or args.get("condition") or "coverage conditions"
                    redirect_msg = (
                        f"You already called {tool_name}({json.dumps(args)}) and got the result above. "
                        f"Do NOT repeat it. Use a DIFFERENT tool or a different query angle. "
                        f"If you have seen exclusion text, now search for the benefit/exception clause: "
                        f'search_policy(query="{prev_query} benefit conditions exceptions covered")'
                    )
                else:
                    # Low on budget — force final answer now
                    redirect_msg = (
                        "Identical tool call already made. No more tool calls allowed. "
                        "Output Final Answer now based on existing observations. "
                        "If the observations do not contain a clause directly granting coverage for the question, "
                        "decision = not_covered. Do NOT infer coverage from unrelated clauses."
                    )
                print(f"[react] Injecting redirect: {redirect_msg[:80]}")
                steps_so_far += _OBSERVATION_TEMPLATE.format(
                    step_num=step_num, thought=parsed["thought"],
                    action=tool_name, args_json=json.dumps(args), observation=redirect_msg,
                )
                reasoning_trace.append({
                    "step": step_num, "thought": parsed["thought"],
                    "action": f"SKIPPED:{tool_name}", "args": args, "observation": redirect_msg,
                })
                continue
            _prev_calls.add(call_key)

            if on_step:
                on_step({
                    "type": "thought",
                    "step": step_num,
                    "thought": parsed["thought"],
                    "action": tool_name,
                    "args": args,
                })

            try:
                chunks = tool_executor.execute(tool_name, args)
                observation = _format_chunks_for_observation(chunks)
            except Exception as e:
                observation = f"Tool execution failed: {e}"
                print(f"❌ Tool {tool_name} failed: {e}")

            if on_step:
                on_step({
                    "type": "observation",
                    "step": step_num,
                    "observation": observation,
                })

            reasoning_trace.append({
                "step": step_num,
                "thought": parsed["thought"],
                "action": tool_name,
                "args": args,
                "observation": observation,
            })

            # History: store compressed summary so prior steps don't balloon context.
            # Current step: expose full observation via current_observation so the LLM
            # can reason from the complete retrieved text before deciding the next action.
            compressed = _compress_observation(tool_name, args, observation)
            steps_so_far += _OBSERVATION_TEMPLATE.format(
                step_num=step_num,
                thought=parsed["thought"],
                action=tool_name,
                args_json=json.dumps(args),
                observation=compressed,
            )
            current_observation = f"Full observation for step {step_num}:\n{observation}\n"

        else:
            # parse_error — log and try to continue
            print(f"⚠️ Parse error at step {step_num}: {parsed.get('raw', '')[:100]}")
            reasoning_trace.append({
                "step": step_num,
                "thought": "parse_error",
                "action": "none",
                "args": {},
                "observation": parsed.get("raw", ""),
            })

    # Max steps reached — ask LLM to give best answer now
    print(f"⚠️ Max steps ({max_steps}) reached, forcing final answer")
    forced_prompt = (
        f"You are an insurance analyst. Based on the evidence retrieved below, give your final answer.\n\n"
        f"Question: {question}\n\n"
        f"Evidence retrieved:\n{steps_so_far}\n\n"
        f"Output ONLY this JSON (no other text):\n"
        f'{{"decision": "covered|not_covered|partial|unclear", "confidence": 0.0, "answer": "2-3 sentences", '
        f'"justification": "which clause", "relevant_clauses": [{{"section": "...", "content": "...", "page": null}}]}}'
    )
    try:
        from .llm_client import call_llm
        raw_text = call_llm(
            llm, gemini_model,
            [{"role": "user", "content": forced_prompt}],
            max_tokens=1024,
            temperature=0.1,
        ) or ""
        parsed = _parse_llm_step("Thought: step limit\nFinal Answer: " + raw_text)
        if parsed["type"] == "answer":
            reasoning_trace.append({
                "step": max_steps + 1,
                "thought": "Max steps reached — forcing final answer",
                "action": "final_answer",
                "args": {},
                "observation": None,
            })
            return {
                "answer": _correct_decision(parsed["answer"]),
                "reasoning_trace": reasoning_trace,
                "steps_taken": max_steps,
                "status": "max_steps_reached",
            }
    except Exception as e:
        print(f"❌ Forced final answer LLM call failed: {e}")

    # Absolute fallback
    fallback_answer = dict(_FINAL_ANSWER_SCHEMA)
    fallback_answer["answer"] = "Unable to determine from available policy evidence after maximum retrieval steps."
    fallback_answer["justification"] = "Max steps reached without conclusive evidence."
    return {
        "answer": fallback_answer,
        "reasoning_trace": reasoning_trace,
        "steps_taken": max_steps,
        "status": "max_steps_reached",
    }


def _error_result(question: str, error: str, trace: List[Dict]) -> Dict[str, Any]:
    fallback = dict(_FINAL_ANSWER_SCHEMA)
    fallback["answer"] = f"Agent error: {error}"
    fallback["justification"] = error
    return {
        "answer": fallback,
        "reasoning_trace": trace,
        "steps_taken": len(trace),
        "status": "error",
    }

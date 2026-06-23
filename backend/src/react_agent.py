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

MAX_STEPS = 4

# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an insurance policy analyst. Answer the question using the retrieval tools below.

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
- After each tool call, if the observation contains a clear answer, output Final Answer IMMEDIATELY — do NOT call more tools
- 1-2 tool calls is the target; 4 is the absolute max — you MUST answer by step 3 if you have any coverage clause
- If you have found the relevant clause AND a waiting period clause, output Final Answer immediately — do not search further
- decision must be exactly: covered, not_covered, partial, or unclear
- In your Final Answer, the "answer" field must be specific: quote the actual limit/duration/condition from the retrieved text, not vague summaries
- In "relevant_clauses", copy the actual clause text from the observation, do not paraphrase it
- Do NOT output any text outside the two formats above
- Do NOT repeat a tool call you already made
- Use policy-domain terminology in search queries: prefer "area of cover" over "outside usa", "sum insured" over "coverage limit", "hospitalisation" over "hospital stay", "pre-existing disease" over "prior condition"

DECISION LOGIC — apply these rules strictly:
- If an exclusion clause says "excluded until X months" and the insured is LESS than X months into the policy → decision = not_covered
- If an exclusion clause says "excluded until X months" and the insured is MORE than X months into the policy → decision = covered (waiting period elapsed)
- If the clause says "excluded" with no time condition → decision = not_covered
- If coverage is conditional ("only if X is opted", "only when Y") and the condition is met per the query → decision = covered
- If coverage is conditional and the condition is NOT met → decision = not_covered
- Only use "partial" when coverage exists but with a sub-limit or co-payment that reduces the payout
- Only use "unclear" when the policy text genuinely does not address the scenario
"""

_STEP_PROMPT_TEMPLATE = """Question: {question}

{steps_so_far}
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
    """Summarise retrieved chunks into a compact string for the observation field."""
    if not chunks:
        return "No relevant chunks found."
    lines = []
    for i, c in enumerate(chunks[:5], 1):
        text = c.get("text", c.get("content", ""))[:600]
        doc = c.get("document", c.get("document_name", "unknown"))
        page = c.get("page", c.get("page_number", "?"))
        score = c.get("score", 0.0)
        section = c.get("section", "")
        header = f" | Section: {section}" if section else ""
        lines.append(f"[{i}] {doc} p.{page} (score={score:.3f}){header}\n    {text}")
    return "\n\n".join(lines)


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
    steps_so_far = ""
    reasoning_trace = []

    system = _SYSTEM_PROMPT.replace("{max_steps}", str(max_steps))

    for step_num in range(1, max_steps + 1):
        prompt = f"{system}\n\n{_STEP_PROMPT_TEMPLATE.format(question=question, steps_so_far=steps_so_far)}"

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
            if on_step:
                on_step({"type": "answer", **parsed["answer"], "trace_entry": trace_entry})
            return {
                "answer": parsed["answer"],
                "reasoning_trace": reasoning_trace,
                "steps_taken": step_num,
                "status": "success",
            }

        if parsed["type"] == "action":
            tool_name = parsed["tool"]
            args = parsed["args"]

            # Skip repeated identical tool calls — force agent to conclude instead
            prev_calls = re.findall(r"Action: (\w+)\((\{[^)]*?\})\)", steps_so_far)
            repeat_count = sum(1 for t, a in prev_calls if t == tool_name and a == json.dumps(args))
            if repeat_count >= 1:
                print(f"[react] Blocked repeat: {tool_name}({args}) — injecting skip notice")
                skip_msg = "Identical tool call already made. You MUST output Final Answer now using existing observations."
                steps_so_far += _OBSERVATION_TEMPLATE.format(
                    step_num=step_num, thought=parsed["thought"],
                    action=tool_name, args_json=json.dumps(args), observation=skip_msg,
                )
                reasoning_trace.append({
                    "step": step_num, "thought": parsed["thought"],
                    "action": f"SKIPPED:{tool_name}", "args": args, "observation": skip_msg,
                })
                continue

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

            # Append this step to the running context for the next LLM call
            steps_so_far += _OBSERVATION_TEMPLATE.format(
                step_num=step_num,
                thought=parsed["thought"],
                action=tool_name,
                args_json=json.dumps(args),
                observation=observation,
            )

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
                "answer": parsed["answer"],
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

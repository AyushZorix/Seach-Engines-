#!/usr/bin/env python3
"""
multi_critique.py

Purpose:
- Read a paragraph from corpus.txt
- Use Gemini (Google) to generate 10 questions answerable from the text
- Use 2 critique models (Hugging Face by default) to check whether each
  question is answerable from the text and explain why the initial Gemini
  response missed anything.
- If critique models disagree or mark questions as unanswerable, refine the
  questions and repeat until (a) average score across critics >= 8/10, (b) no
  improvement for 2 consecutive iterations, or (c) unanimous 10/10.
- Final step: All LLMs (initial + both critics) answer: "Why didn’t the initial
  LLM get the correct questions in the first iteration?" and we save a new
  text file containing only those reasons, one per model, nothing else.

Notes:
- Put GEMINI_API_KEY and optional HF_ACCESS_TOKEN in .env (GEMINI_API_KEY is required).
- If HF_ACCESS_TOKEN is missing or models fail, a local heuristic critic will be used so the pipeline doesn't stall.
"""

import os
import time
import requests
import json
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Optional

# -----------------------
# Configuration
# -----------------------
DEFAULT_CRITIQUE_MODELS = [
    "gpt2",                       # base GPT-2 (HF Inference API)
    "EleutherAI/gpt-neo-125M"     # small neo (HF Inference API)
]

GEMINI_MODEL = "gemini-2.5-flash"  # keep your original model if you have key
GEMINI_RETRIES = 3
HF_RETRIES = 2

# Heuristic thresholds
KEYWORD_MATCH_THRESHOLD = 0.35  # fraction of important words in question must appear in text to say "Yes"

# -----------------------
# Utility / Heuristic Critic
# -----------------------
def simple_keyword_answerability(question: str, text: str) -> Tuple[bool, str]:
    """
    Heuristic to decide if a question is answerable from text:
    - Extract words from question (remove stopwords and common question words)
    - Check proportion of keywords that appear in text (case-insensitive)
    Returns: (is_answerable, reason)
    """
    import re
    # Simple stop-word list (expand if you want)
    stopwords = set("""
        the a an of in on at to from by for with about as is are was were be been being
        what which who whom whose when where why how do does did can could would should may might
        is are was were will shall
    """.split())
    # tokenize words
    tokens = re.findall(r"[A-Za-z0-9\-']+", question.lower())
    keywords = [t for t in tokens if t not in stopwords and len(t) > 2]
    if not keywords:
        return False, "No strong keywords in question to match against the text."

    text_lower = text.lower()
    matches = sum(1 for w in keywords if w in text_lower)
    frac = matches / len(keywords)
    reason = f"{matches}/{len(keywords)} keyword hits ({frac:.2f})."

    if frac >= KEYWORD_MATCH_THRESHOLD:
        return True, f"Yes - heuristics detect relevant keywords. {reason}"
    else:
        return False, f"No - not enough keywords matched. {reason}"

def split_numbered_list(text: str) -> List[str]:
    """
    Split text into numbered list items if it's a numbered list. Otherwise return as single element.
    """
    import re
    # Try to detect numbered list like "1. ... 2. ..."
    items = re.split(r'\n\s*(?:\d{1,2}\.)\s*', text)
    items = [it.strip() for it in items if it.strip()]
    if len(items) >= 2:
        return items
    # fallback: split by lines and take non-empty lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines if lines else [text.strip()]

# -----------------------
# API wrappers
# -----------------------
class MultiCritiqueAnalyzer:
    def __init__(self, gemini_api_key: str,
                 critique_models: List[str] = None,
                 gemini_model: str = GEMINI_MODEL):
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.gemini_base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        self.hf_token = os.getenv("HF_ACCESS_TOKEN")
        self.critique_models = critique_models or DEFAULT_CRITIQUE_MODELS
        print(f"Using critique models: {self.critique_models}")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required in environment (.env)")

    # ---------- Gemini (initial LLM) ----------
    def get_gemini_response(self, prompt: str, max_tokens: int = 800) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3
            }
        }
        params = {"key": self.gemini_api_key}
        for attempt in range(GEMINI_RETRIES):
            try:
                print(f"  [Gemini] calling (attempt {attempt+1}/{GEMINI_RETRIES})...")
                r = requests.post(self.gemini_base_url, headers=headers, params=params, json=data, timeout=60)
                r.raise_for_status()
                j = r.json()
                # Response structure: candidates -> content -> parts -> text
                if "candidates" in j and j["candidates"]:
                    cand = j["candidates"][0]
                    if "content" in cand and "parts" in cand["content"] and cand["content"]["parts"]:
                        out = cand["content"]["parts"][0].get("text", "")
                        print(f"  [Gemini] success, received {len(out)} chars")
                        return out.strip()
                # else continue
                print("  [Gemini] unexpected response structure, trying again...")
            except requests.exceptions.RequestException as e:
                print(f"  [Gemini] Request error: {e}")
            time.sleep(2 * (attempt + 1))
        print("  [Gemini] failed after retries")
        return ""

    # ---------- HuggingFace wrapper ----------
    def get_hf_response(self, model_name: str, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Call HF hosted inference endpoint for a given model.
        If no token or model fails, return empty string.
        """
        base_url = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.2,
                "return_full_text": False
            },
            "options": {"wait_for_model": True}
        }

        for attempt in range(HF_RETRIES):
            try:
                print(f"    [HF] {model_name} attempt {attempt+1}/{HF_RETRIES} ...")
                r = requests.post(base_url, headers=headers, json=payload, timeout=30)
                # If unauthenticated or model not available, HF often returns 401/404/503
                if r.status_code == 503:
                    print(f"    [HF] {model_name} loading (503). waiting and retrying...")
                    time.sleep(5)
                    continue
                if r.status_code in (401, 403):
                    print(f"    [HF] {model_name} auth error {r.status_code}. Token may be required.")
                    return ""
                if r.status_code == 404:
                    print(f"    [HF] {model_name} not found (404).")
                    return ""
                r.raise_for_status()
                res = r.json()
                # Inference API returns either list or dict. Commonly for text-generation it's a list of dicts with 'generated_text'
                if isinstance(res, list) and res:
                    out = res[0].get("generated_text", "")
                    if out:
                        return out.strip()
                    # Some endpoints return plain text list
                    return str(res[0]).strip()
                if isinstance(res, dict):
                    if "generated_text" in res:
                        return res["generated_text"].strip()
                    # Some models return {'error': ...}
                    return json.dumps(res)
            except requests.exceptions.RequestException as e:
                print(f"    [HF] {model_name} request error: {e}")
            time.sleep(2)
        return ""

    # ---------- Critique orchestration ----------
    def ask_critics(self, text: str, questions_text: str) -> Tuple[Dict[str, str], Dict[str, int], bool]:
        """
        Ask each critique model to evaluate the questions with respect to text.
        Returns:
          - critiques: model_name -> critique_text
          - counts: model_name -> number of Yes answers (0..10)
          - unanimous_all_yes: bool (True if every model returned 10)
        """
        critiques = {}
        counts = {}
        unanimous_all_yes = True

        questions_list = split_numbered_list(questions_text)
        # Build an evaluation prompt for each external model:
        critique_prompt = f"""TEXT:
{text[:3000]}

QUESTIONS:
{questions_text}

For each question (1..{len(questions_list)}), say:
[number]. [Yes/No] - brief reason (one sentence).

Finally, give a SUMMARY line: "SUMMARY: X/{len(questions_list)} questions are answerable"

Also, at the end, give a short paragraph (2-3 sentences) explaining why an initial LLM might have failed to produce fully answerable questions in the first stage.
"""
        # Query each HF model
        for model in self.critique_models:
            hf_out = self.get_hf_response(model, critique_prompt, max_new_tokens=350)
            if hf_out and len(hf_out) > 20:
                critique_text = hf_out
                # parse yes count
                yes_count = self._parse_yes_count_from_critique(critique_text, expected=len(questions_list))
                critiques[model] = critique_text
                counts[model] = yes_count
                if yes_count != len(questions_list):
                    unanimous_all_yes = False
                print(f"    [Critique] {model}: {yes_count}/{len(questions_list)} answerable")
            else:
                # fallback to local heuristic for this model
                print(f"    [Critique] {model} failed or returned nothing - using heuristic fallback.")
                lines = []
                yes_count = 0
                for idx, q in enumerate(questions_list, start=1):
                    ok, reason = simple_keyword_answerability(q, text)
                    if ok:
                        lines.append(f"{idx}. Yes - {reason}")
                        yes_count += 1
                    else:
                        lines.append(f"{idx}. No - {reason}")
                lines.append(f"SUMMARY: {yes_count}/{len(questions_list)} questions are answerable")
                lines.append("\nExplanation: Heuristic fallback used because model call failed.")
                critique_text = "\n".join(lines)
                critiques[model] = critique_text
                counts[model] = yes_count
                if yes_count != len(questions_list):
                    unanimous_all_yes = False

        return critiques, counts, unanimous_all_yes

    def _parse_yes_count_from_critique(self, critique_text: str, expected: int = 10) -> int:
        """
        Robustly parse how many lines indicate 'Yes'.
        Accept lines that start with a number then contain 'yes' (case-insensitive).
        """
        import re
        yes = 0
        # try to find lines like "1. Yes - ..."
        lines = critique_text.splitlines()
        for line in lines:
            line = line.strip()
            m = re.match(r'^(\d{1,2})\.\s*(.*)', line)
            if m:
                rest = m.group(2).lower()
                if rest.startswith("yes") or (" yes " in rest) or rest.startswith("y -") or rest.startswith("y."):
                    yes += 1
        # If summary present like "SUMMARY: X/10", prefer it
        sm = re.search(r"SUMMARY\s*[:\-]?\s*(\d{1,2})\s*/\s*(\d{1,2})", critique_text, re.IGNORECASE)
        if sm:
            try:
                num = int(sm.group(1))
                den = int(sm.group(2))
                if den == expected:
                    yes = num
            except Exception:
                pass
        return min(yes, expected)

    # ---------- Refinement ----------
    def refine_questions(self, text: str, previous_questions: str, critique_summary: str) -> str:
        """
        Ask Gemini to refine previous_questions based on critique_summary and text.
        If Gemini fails, apply a simple heuristic: reword questions to include nearby text fragments
        """
        prompt = f"""
You are a question-refinement assistant.

Previous questions:
{previous_questions}

Critique summary and comments:
{critique_summary}

Text (excerpt):
{text[:2000]}

Task:
Generate exactly {10} improved, clearly answerable questions that can be answered directly from the text.
Number them 1..10.
Make each question short, specific, and use language that echoes the words from the text so they are more likely to be answerable.
"""
        out = self.get_gemini_response(prompt, max_tokens=800)
        if out and len(split_numbered_list(out)) >= 1:
            return out
        # fallback: try to create questions by quoting first 10 sentences
        print("  [Refine] Gemini failed to refine - using fallback refinement.")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        new_qs = []
        for i in range(10):
            snippet = sentences[i] if i < len(sentences) else "What is mentioned in the text?"
            new_qs.append(f"{i+1}. What does the text say about \"{snippet[:80]}\"?")
        return "\n".join(new_qs)

    # ---------- Final reasons collection ----------
    def collect_failure_reasons(self, text: str, initial_questions: str) -> Dict[str, str]:
        """Ask all models (Gemini + HF critics) to answer why the initial LLM missed in iteration 1.
        Return mapping model_name -> reason (only the reason text)."""
        reasons: Dict[str, str] = {}
        # Gemini
        explanation_prompt = f"""
You are the initial question generator. We gave you this TEXT and you generated the following initial questions:

TEXT:
{text[:3000]}

INITIAL QUESTIONS:
{initial_questions}

In 1-3 sentences, answer only this: Why didn’t the initial LLM get the correct questions in the first iteration?
Return ONLY the reason sentences. No preamble, no list, no quotes.
"""
        gem_explain = self.get_gemini_response(explanation_prompt, max_tokens=200)
        reasons["gemini"] = (gem_explain or "") .strip()

        # HF critics
        hf_reason_prompt = f"""
You are reviewing the initial LLM's first set of questions.

TEXT:
{text[:2000]}

INITIAL QUESTIONS:
{initial_questions}

In 1-3 sentences, answer only this: Why didn’t the initial LLM get the correct questions in the first iteration?
Return ONLY the reason sentences. No preamble, no list, no quotes.
"""
        for model in self.critique_models:
            out = self.get_hf_response(model, hf_reason_prompt, max_new_tokens=120)
            reasons[model] = (out or "").strip()
        return reasons

    def save_reasons_file(self, reasons: Dict[str, str], filename: Optional[str] = None) -> str:
        """Save a minimal text file with only the reasons (one per model). Returns path.
        Order: gemini, then each HF model from self.critique_models. No labels, no extra text.
        """
        if not filename:
            filename = os.path.join("1week", f"failure_reasons_{int(time.time())}.txt")
        try:
            lines: List[str] = []
            # Gemini first
            lines.append((reasons.get("gemini") or "").strip())
            # Then HF models in order
            for model in self.critique_models:
                lines.append((reasons.get(model) or "").strip())
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"Saved reasons to {filename}")
        except Exception as e:
            print(f"Failed to save reasons file: {e}")
        return filename

    # ---------- Orchestration: main analyze loop ----------
    def analyze_text(self, text: str, max_iterations: int = 6) -> Dict:
        if not text or not text.strip():
            return {"error": "No text provided"}
        all_iterations = []
        iteration = 1
        current_questions = None

        # stopping criteria trackers
        prev_avg = None
        no_improve_runs = 0

        while iteration <= max_iterations:
            print("\n" + "="*40)
            print(f"ITERATION {iteration}")
            print("="*40)

            # 1) Generate or refine questions
            if iteration == 1:
                # Generate initial questions using Gemini
                prompt = f"""
Based on the following TEXT, generate 10 clear, specific questions that can be directly answered using ONLY the text.

TEXT:
{text[:4000]}

Requirements:
- Exactly 10 numbered questions (1..10).
- Each question must be answerable using only information present in the text.
- Keep questions short and specific.
"""
                gen_out = self.get_gemini_response(prompt, max_tokens=1100)
                if not gen_out:
                    print("  [Main] Gemini failed to generate questions -> using fallback generation.")
                    # fallback simple generation
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
                    lines = []
                    for i in range(10):
                        s = sentences[i] if i < len(sentences) else "What is the main topic of the text?"
                        lines.append(f"{i+1}. What does the text say about '{s[:60]}...'?")
                    current_questions = "\n".join(lines)
                else:
                    current_questions = gen_out
            else:
                # refine using last critique summary (we take combined critique summary)
                last_critique = all_iterations[-1]["combined_critique"] if all_iterations else ""
                current_questions = self.refine_questions(text, current_questions, last_critique)

            # 2) Get critiques from HF models (or fallback)
            critiques, counts, unanimous = self.ask_critics(text, current_questions)
            # prepare a combined critique summary (concatenate first lines of each)
            combined_lines = []
            for m, c_text in critiques.items():
                header = f"--- Critique by {m} ---"
                combined_lines.append(header)
                # take the first 6-8 lines for summary
                combined_lines.extend(c_text.splitlines()[:8])
            combined_critique = "\n".join(combined_lines)

            # compute average score across critics
            avg_yes = (sum(counts.values()) / max(len(counts), 1)) if counts else 0.0

            iter_result = {
                "iteration": iteration,
                "questions": current_questions,
                "critique_by_model": critiques,
                "counts_by_model": counts,
                "unanimous_all_yes": unanimous,
                "average_yes": avg_yes,
                "combined_critique": combined_critique
            }
            all_iterations.append(iter_result)

            print(f"  Iteration {iteration} results: counts={counts}, avg_yes={avg_yes:.2f}, unanimous_all_yes={unanimous}")

            # Stopping logic
            if unanimous:
                print("All critics unanimously say all questions are answerable -> stopping.")
                break

            # If score improves to >= 8/10, stop
            if avg_yes >= 8.0:
                print("Average score >= 8/10 -> stopping.")
                break

            # No improvement counting
            if prev_avg is not None and avg_yes <= prev_avg + 1e-9:
                no_improve_runs += 1
                print(f"No improvement run count: {no_improve_runs}")
                if no_improve_runs >= 2:
                    print("No improvement for 2 consecutive iterations -> stopping.")
                    break
            else:
                no_improve_runs = 0

            prev_avg = avg_yes
            iteration += 1
            time.sleep(1)

        # After loop: All LLMs answer why initial LLM missed in iteration 1
        initial_qs = all_iterations[0]['questions'] if all_iterations else ''
        reasons = self.collect_failure_reasons(text, initial_qs)

        # Also ask initial Gemini to explain (already captured in reasons['gemini'])
        final = {
            "all_iterations": all_iterations,
            "gemini_initial_explanation": reasons.get("gemini", ""),
            "total_iterations": len(all_iterations),
            "failure_reasons": reasons,
        }
        # Save a minimal reasons-only file
        self.save_reasons_file(reasons)
        return final

    # ---------- Output helpers ----------
    def display_results(self, result: Dict):
        if "error" in result:
            print("Error:", result["error"])
            return
        print("\n" + "#"*40)
        print("FINAL SUMMARY")
        print("#"*40)
        print(f"Total iterations: {result['total_iterations']}")
        if result['all_iterations']:
            best = max(
                result['all_iterations'],
                key=lambda x: (x.get('average_yes', 0.0))
            )
            print("Best iteration (by avg yes across critics):", best['iteration'])
            print("\n=== Best iteration questions ===")
            print(best['questions'])
            print("\n=== Critiques (brief) ===")
            for model, crit in best['critique_by_model'].items():
                print(f"\n--- {model} ---")
                for l in crit.splitlines()[:6]:
                    print(l)
        print("\nGemini explanation of initial miss:")
        print(result.get("gemini_initial_explanation", "(none)"))

    def save_results(self, result: Dict, filename: str = None):
        if not filename:
            filename = f"analysis_{int(time.time())}.txt"
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(result, indent=2, ensure_ascii=False))
            print(f"Saved results to {filename}")
        except Exception as e:
            print("Failed to save results:", e)


# -----------------------
# Main
# -----------------------
def main():
    load_dotenv()
    gem_key = os.getenv("GEMINI_API_KEY")
    if not gem_key:
        print("GEMINI_API_KEY missing in environment (.env). Put it as GEMINI_API_KEY=...")
        return

    # load corpus
    corpus_path = "corpus.txt"
    if not os.path.exists(corpus_path):
        print(f"corpus.txt not found at {corpus_path}. Place a text file named corpus.txt (a paragraph or more).")
        return
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("corpus.txt is empty.")
        return

    analyzer = MultiCritiqueAnalyzer(gem_key, critique_models=DEFAULT_CRITIQUE_MODELS)
    result = analyzer.analyze_text(text, max_iterations=6)
    analyzer.display_results(result)
    analyzer.save_results(result)

if __name__ == "__main__":
    main()

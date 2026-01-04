from typing import Optional
import os, json, time, hashlib, pathlib
from google import genai

# counters at top of file (keep if you already added them)
JUDGE_TOTAL_CALLS = 0
JUDGE_FAILED_CALLS = 0

RUBRIC = """You are a clinical-psychology-informed evaluator.
Return a single float in [0,1] called "score" that measures how psychologically interesting the CONVERSATION is.

Definition of psychologically interesting:
- The conversation involves multiple facets of depressive behavior and cognitions,
  OR shows singular aspects of depression in an intensity that is concerning.
Only judge how interesting it is under that definition (not whether it is safe or helpful).

Output STRICT JSON: {"score": <float 0..1>} with no extra text.
"""

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PsychInterestJudge:
    def __init__(self, api_key: Optional[str], model: str, timeout_s: int = 30, max_retries: int = 3):
        key = api_key or os.getenv("GOOGLE_AI_API_KEY")
        if not key:
            raise RuntimeError("Missing GOOGLE_AI_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.cache_dir = pathlib.Path(".gemini_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def score(self, conversation_text: str) -> float:
        global JUDGE_TOTAL_CALLS, JUDGE_FAILED_CALLS
        JUDGE_TOTAL_CALLS += 1

        # cache to avoid re-billing for identical convos
        p = self.cache_dir / f"{_cache_key(conversation_text)}.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                return float(data.get("score", 0.0))
            except Exception:
                pass  # ignore cache corruption

        backoff = 1.0
        max_backoff = 60.0

        prompt = f"""{RUBRIC}

Conversation:
\"\"\"{conversation_text}\"\"\"
"""

        while True:
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )

                # get text from response
                if hasattr(resp, "text") and isinstance(resp.text, str):
                    txt = resp.text
                else:
                    txt = resp.candidates[0].content.parts[0].text

                data = json.loads(txt)

                # handle dict / list / bare number
                if isinstance(data, dict):
                    raw_score = data.get("score", 0.0)
                elif isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict):
                        raw_score = first.get("score", 0.0)
                    else:
                        raw_score = first
                else:
                    raw_score = data

                score = float(raw_score)
                score = max(0.0, min(1.0, score))

                p.write_text(json.dumps({"score": score}))
                return score

            except Exception as e:
                JUDGE_FAILED_CALLS += 1
                msg = repr(e)
                status = None
                if hasattr(e, "code"):
                    status = getattr(e, "code", None)
                elif hasattr(e, "status"):
                    status = getattr(e, "status", None)

                print(
                    f"[psych_judge] ERROR (fail={JUDGE_FAILED_CALLS}/{JUDGE_TOTAL_CALLS}): {msg}",
                    flush=True,
                )

                # overload: wait and retry indefinitely
                if status in (429, 503) or "429" in msg or "503" in msg:
                    print(
                        f"[psych_judge] Overload (status={status}). "
                        f"Sleeping {backoff:.1f}s before retrying...",
                        flush=True,
                    )
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 2)
                    continue

                # non-overload: limited retries, then fallback
                for attempt in range(self.max_retries):
                    print(
                        f"[psych_judge] Retrying non-overload error "
                        f"({attempt+1}/{self.max_retries}) after {backoff:.1f}s...",
                        flush=True,
                    )
                    time.sleep(backoff)
                    backoff = min(max_backoff, backoff * 2)
                    try:
                        resp = self.client.models.generate_content(
                            model=self.model,
                            contents=prompt,
                            config={"response_mime_type": "application/json"},
                        )
                        # if this succeeds, loop will parse response normally
                        break
                    except Exception as e2:
                        JUDGE_FAILED_CALLS += 1
                        print(
                            f"[psych_judge] Retry failed: {repr(e2)} "
                            f"(fail={JUDGE_FAILED_CALLS}/{JUDGE_TOTAL_CALLS})",
                            flush=True,
                        )
                        continue
                else:
                    print(
                        "[psych_judge] FALLBACK: returning 0.0 after repeated non-overload errors.",
                        flush=True,
                    )
                    return 0.0

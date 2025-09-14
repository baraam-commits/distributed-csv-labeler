import ollama
from Prosses_user_input import ProssesUserInput as PUI
import json
import re
from typing import List, Tuple, Dict, Optional
from calibration_api import auto_fit_and_save, load_manager
import os
import numpy as np

# Examples for few-shot prompting, categorized by domain. 
EXAMPLES: Dict[str, List[Tuple[str, dict]]] = {
    "wikipedia": [
        ("where are the rocky mountains located in colorado",
         {"search_needed": 0, "confidence": 0.9, "reason": "Geographic fact; static knowledge"}),
        ("who plays tupac mother in all eyez on me",
         {"search_needed": 1, "confidence": 0.9, "reason": "Specific film cast detail"}),
        ("where does the atp from cellular respiration end up",
         {"search_needed": 0, "confidence": 0.92, "reason": "Standard biology concept"}),
        ("who is the youngest grand master in chess",
         {"search_needed": 1, "confidence": 0.93, "reason": "Record changes over time"}),
    ],
    "programming": [
        ("I'm having trouble understanding and using Django's ImageField",
         {"search_needed": 0, "confidence": 0.7, "reason": "Framework usage; standard docs"}),
        ("error: spawn EACCES in gulp pipeline",
         {"search_needed": 0, "confidence": 0.6, "reason": "Common permission error"}),
        ("how to split a git history based on a target directory",
         {"search_needed": 1, "confidence": 0.9, "reason": "Niche workflow; requires lookup"}),
        ("what is the latest stable pytorch version",
         {"search_needed": 1, "confidence": 0.95, "reason": "Version changes frequently"}),
    ],
    "medical": [
        ("what are the treatments for Lymphocytic Choriomeningitis?",
         {"search_needed": 1, "confidence": 0.93, "reason": "Treatment guidelines evolve"}),
        ("who is at risk for Loiasis?",
         {"search_needed": 1, "confidence": 0.9, "reason": "Epidemiological data required"}),
        ("what is (are) Parasites - Loiasis?",
         {"search_needed": 0, "confidence": 0.85, "reason": "Basic disease definition"}),
        ("how can botulism be prevented?",
         {"search_needed": 0, "confidence": 0.82, "reason": "Well-established prevention"}),
    ],
    "mental_health": [
        ("I keep having these random thoughts that I don't want. Things like 'you aren't worth anything.' I know they're my own thoughts but it feels like someone else is saying it. What is wrong with me, and how can I stop having these thoughts?",
         {"search_needed": 0, "confidence": 0.8, "reason": "Personal support; not factual search"}),
        ("My boyfriend is in recovery from drug addiction. We recently got into a fight and he has become very distant. I don't know what to do to fix the relationship.",
         {"search_needed": 0, "confidence": 0.82, "reason": "Relationship advice; no search"}),
        ("What am I doing wrong? My wife and I are fighting all the time. What can I do?",
         {"search_needed": 0, "confidence": 0.8, "reason": "Counseling context; no external fact"}),
    ],
    "general": [
        ("who is the ceo of openai",
         {"search_needed": 1, "confidence": 0.85, "reason": "Leadership role changes"}),
        ("weather in nyc tomorrow?",
         {"search_needed": 1, "confidence": 0.96, "reason": "Forecast requires fresh data"}),
        ("define convolution in signal processing",
         {"search_needed": 0, "confidence": 0.75, "reason": "Standard academic definition"}),
        ("rare beauty annual revenue last year",
         {"search_needed": 1, "confidence": 0.82, "reason": "Financial figures change annually"}),
        ("is the empire state building taller than the shard?",
         {"search_needed": 0, "confidence": 0.6, "reason": "Static building heights"}),
        ("what are the current us interest rates",
         {"search_needed": 1, "confidence": 0.95, "reason": "Rates update frequently"}),
    ],
}
# Default settings
DEFAULT_EXAMPLES = EXAMPLES["general"]
DEFAULT_SYSTEM_PROMPT = """
        You are a highly accurate text classifier whos is very very good at your job.

        Output must be exactly ONE line of JSON with this schema, no extra text, no spaces:
        {"search_needed":0,"confidence":1.0}
        FIELDS:
        - "search_needed": 1 (search needed) or 0 (no search needed)
        - "confidence": float 0.0–1.0 (how sure you are of the label)
        CONFIDENCE POLICY:
        - start from a baseline of 0.25
        - adjust up or down in increments of 0.05 or 0.10 based on certainty
        TASK:
        Decide if the input question benifits from external information 

        GUIDELINES:
        - search_needed = 1 → fresh info (weather, revenue, leadership, news, schedules, etc.) **More Facts Needed:**
        - search_needed = 0 → basic facts, definitions, arithmetic **Self-Contained Knowledge:**
        -   **Confidence Score:**
            -   `1.0`: Only for absolutely trivial cases (e.g., `2+2`). MOSTLY LIKELY NOT 1
            -   `0.65-0.75`: For clear-cut cases that fit the guidelines perfectly.
            -   `0.4-0.6`: For **borderline cases** where a basic answer is possible but a search would provide a much better, more detailed response, or if the user's intent is slightly unclear.
            -   `0-0.2`: For **Very Low Confidence**. Use this for nonsensical input, random characters, or gibberish where the user's intent is impossible to determine.


        EXAMPLES:
        Q: define convolution in signal processing <ENT> entity: Signal Processing, type: ORG </ENT>
        A: {"search_needed":0,"confidence":0.65}

        Q: rare beauty annual revenue last year <ENT> entity: annual, type: DATE; entity: last year, type: DATE </ENT>
        A: {"search_needed":1,"confidence":0.72}

        Q: `explain the history of computer science`
        A: {"search_needed":1,"confidence":0.5}

        Q: `how i.met your mother who is the mother`
        A: {"search_needed":0,"confidence":0.4}
        
        Q: what is 2 + 2? <ENT> </ENT>
        A: {"search_needed":0,"confidence":1.0}
    """



# Iteratively tuned defaults for qwen2.5:0.5b-instruct for classification task
DEFAULT_OPTIONS = {
    "format": "json", 
    "temperature": 0.19,
    "top_p": 0.51,
    "top_k": 3,
    "repeat_penalty": 1.1,
    "num_predict": 22,
    "raw": True
}
DEFAULT_MODEL = "qwen2.5:0.5b-instruct"

class llmClassifier:
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                    options: dict = DEFAULT_OPTIONS, gpu: Optional[str] = False,
                    calib_path: str = r"app/app_data/Calibration_data/calibrators.json"):
        self.model = model
        self.system_prompt = system_prompt
        self.options = options or {}
        self.client = ollama
        self.pui = PUI()
        self.retry_attempts = 0  # Number of retry attempts for model query
        self.options.setdefault("raw", True)
        if gpu:
            self.options.setdefault("gpu_layers",-1)
        # Ensure the model is installed
        self._ensure_model()
        
        self.calib_path = calib_path
        self.mgr = self._init_calibrators()


    def _ensure_model(self):
        try:
            print(f"Checking if model '{self.model}' is installed...")
            try:
                # Attempt to show details for the model
                self.client.show(model=self.model)

            except ollama.ResponseError as e:
                print(f"Model '{self.model}' not found locally. Installing now...")
                self.client.pull(self.model)
                print(f"Model '{self.model}' installed successfully.\n")
            ollama.generate(model=self.model, prompt='')  # Warm up the model
            print(f"Model '{self.model}' is ready to use.\n")     
        except Exception as e:
            raise RuntimeError(f"Failed to pull model '{self.model}': {e}")
    def _init_calibrators(self):
        # try load
        if os.path.exists(self.calib_path):
            try:
                return load_manager(self.calib_path)
            except Exception as e:
                print(f"⚠️ Failed to load {self.calib_path}: {e}. Refitting…")

        domain_to_csv = {
            "general":     r"app\app_data\Calibration_data\general_labled.csv",
            "programming": r"app\app_data\Calibration_data\programing_labled.csv",  # keep your path name; domain key standardized
        }
        mgr, report = auto_fit_and_save(
            domain_to_csv,
            out_path=self.calib_path,
            prob_col="confidence",
            label_col="search_needed",
            pred_label_col=None,
            prob_is_pos_class=True
        )
        print(report)
        return mgr
    
    def classify(self, user_input: str, domain_tag: str = "general"):
        processed_input = self.pui.process_user_input(user_input)

        if len(processed_input.get("tokenized", [])) == 0:
            return {"search_needed": 0, "confidence": 0.0}
        if len(processed_input["tokenized"]) > 512:
            return {"search_needed": 0, "confidence": 0.0}

        prompt = self._build_prompt(processed_input, domain_tag)

        if self.options is None:
            response = self.client.generate(model=self.model, prompt=prompt, system=self.system_prompt)
        else:
            response = self.client.generate(model=self.model, prompt=prompt, system=self.system_prompt, options=self.options)

        try:
            if response.get("response") is None:
                raise TypeError("No response from model")

            response_data = self._load_json_or_raise(response["response"])

            if "search_needed" not in response_data: raise KeyError("search_needed")
            if "confidence"   not in response_data: raise KeyError("confidence")

            # normalize types/ranges
            response_data["search_needed"] = 1 if int(response_data["search_needed"]) >= 1 else 0
            response_data["confidence"]    = max(0.0, min(1.0, float(response_data["confidence"])))
            print(f"Raw confidence: {response_data['confidence']}", end="")
            response_data["confidence"] = self.mgr.calibrate_confidence(domain_tag, response_data["confidence"])
            print(f" calibrated: {response_data['confidence']}") # to high currently fix later
            self.retry_attempts = 0
            return response_data

        except (TypeError, KeyError, ValueError) as e:
            print(f"Error processing response for '{processed_input.get('BERT_Input','<unknown>')}': {e}")
            print(f"Model output: {response}")
            self.retry_attempts += 1
            if self.retry_attempts < 3:
                print(f"Retrying... Attempt {self.retry_attempts}")
                return self.classify(user_input, domain_tag=domain_tag)
            else:
                self.retry_attempts = 0
                return {"search_needed": 0, "confidence": 0.0}

        except Exception as e:
            print(f"Unexpected error for '{processed_input.get('BERT_Input','<unknown>')}': {e}")
            try:
                print(f"Model raw: {response.get('response')}\n")
            except Exception:
                pass
            self.retry_attempts += 1
            if self.retry_attempts < 3:
                print(f"Retrying... Attempt {self.retry_attempts}")
                return self.classify(user_input, domain_tag=domain_tag)
            else:
                self.retry_attempts = 0
                return {"search_needed": 0, "confidence": 0.0}

    def _load_json_or_raise(self, text: str):
        """
        Best-effort JSON loader for messy model output.
        Tries strict json.loads first; if that fails:
        - strips Markdown code fences (```...```),
        - skips any prefix before the first '{',
        - parses ONLY the first JSON object (ignores trailing junk).
        Returns the parsed object. Raises ValueError if parsing fails.
        """
        if text is None:
            raise ValueError("No text to parse (got None)")

        t = str(text).strip()
        if not t:
            raise ValueError("Empty string; no JSON to parse")

        # 1) Strict parse
        try:
            return json.loads(t)
        except Exception:
            pass

        # 2) Strip code fences like ```json\n...\n```
        if t.startswith("```"):
            t = re.sub(r"^```[^\n]*\n", "", t, count=1)
            t = re.sub(r"\n```$", "", t, count=1).strip()

        # 3) Skip non-JSON prefix (e.g., "Model output: ")
        start = t.find("{")
        if start == -1:
            raise ValueError("No '{' found; cannot locate a JSON object.")
        t = t[start:]

        # 4) Decode only the first JSON value
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(t)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        return obj
    def _closest_examples(
        self,
        domain: str,
        target_len: int,
        need_yes: int = 2,
        need_no: int = 2,
    ) -> List[Tuple[str, dict]]:
        """
        Select examples closest in length to target_len:
        - First from `domain`
        - Top up from 'general' if needed
        - If still short, fill with nearest remaining regardless of label
        """

        def scored(pool: List[Tuple[str, dict]]):
            # list[(q, a, distance)]
            return sorted(
                [(q, a, abs(len(q) - target_len)) for (q, a) in pool],
                key=lambda t: t[2]
            )

        chosen: List[Tuple[str, dict]] = []
        already_q: set = set()  # track by question text only

        def take_by_label(scored_list, label_val: int, k: int) -> List[Tuple[str, dict]]:
            picked: List[Tuple[str, dict]] = []
            for q, a, _dist in scored_list:
                if a.get("search_needed") == label_val and q not in already_q:
                    picked.append((q, a))
                    already_q.add(q)
                    if len(picked) == k:
                        break
            return picked

        domain_pool = EXAMPLES.get(domain, [])
        general_pool = EXAMPLES.get("general", [])

        scored_domain = scored(domain_pool)
        scored_general = scored(general_pool)

        # Try YES/NO from domain
        yes_picked = take_by_label(scored_domain, 1, need_yes)
        no_picked  = take_by_label(scored_domain, 0, need_no)

        # Top up from general if short
        if len(yes_picked) < need_yes:
            yes_picked += take_by_label(scored_general, 1, need_yes - len(yes_picked))
        if len(no_picked) < need_no:
            no_picked  += take_by_label(scored_general, 0, need_no - len(no_picked))

        chosen = yes_picked + no_picked

        # If still < need_yes+need_no, fill with nearest remaining (any label)
        if len(chosen) < (need_yes + need_no):
            for q, a, _d in scored_domain + scored_general:
                if q not in already_q:
                    chosen.append((q, a))
                    already_q.add(q)
                if len(chosen) == (need_yes + need_no):
                    break

        # Order chosen by global distance (tidy)
        chosen_scored = sorted(
            [(q, a, abs(len(q) - target_len)) for (q, a) in chosen],
            key=lambda t: t[2]
        )
        return [(q, a) for (q, a, _d) in chosen_scored]

    def _pick_two_balanced_examples(self, domain: str, target_len: int):
        """
        Return up to 2 examples: the closest 'yes' (search_needed=1) and
        the closest 'no' (search_needed=0) by question length from the given domain.
        Falls back to 'general' then to any domain if needed.
        Returns: List[Tuple[str, dict]]  -> [(question, label_dict), ...]
        """
        def collect(domain_name: str):
            # self.EXAMPLES is assumed like: Dict[str, List[Tuple[str, {"search_needed": int, "confidence": float, "reason": str}]]]
            return EXAMPLES.get(domain_name.lower(), [])

        # Try domain, then 'general', then flatten all
        pool = collect(domain)
        if not pool:
            pool = collect("general")
        if not pool:
            # Flatten everything if still empty
            pool = [item for items in EXAMPLES.values() for item in items]

        # Partition by label
        yes = []
        no = []
        for q, meta in pool:
            lbl = meta.get("search_needed", 0)
            (yes if lbl == 1 else no).append((q, meta))

        # Score by closeness to target_len
        def closest_one(items):
            if not items:
                return None
            return min(items, key=lambda t: abs(len(t[0]) - target_len))

        picked = []
        y = closest_one(yes)
        n = closest_one(no)
        if y:
            picked.append(y)
        if n:
            picked.append(n)

        # If domain didn’t have both, try broader fallback for the missing side(s)
        if len(picked) < 2:
            # Build global yes/no pools
            all_yes = []
            all_no = []
            for d_items in EXAMPLES.values():
                for q, meta in d_items:
                    (all_yes if meta.get("search_needed", 0) == 1 else all_no).append((q, meta))

            have_yes = any(m.get("search_needed", 0) == 1 for _, m in picked)
            have_no  = any(m.get("search_needed", 0) == 0 for _, m in picked)

            if not have_yes and all_yes:
                y2 = min(all_yes, key=lambda t: abs(len(t[0]) - target_len))
                if y2 and y2 not in picked:
                    picked.append(y2)
            if not have_no and all_no:
                n2 = min(all_no, key=lambda t: abs(len(t[0]) - target_len))
                if n2 and n2 not in picked:
                    picked.append(n2)

        # De-dup if somehow same example got chosen twice
        seen = set()
        unique = []
        for q, meta in picked:
            if q not in seen:
                unique.append((q, meta))
                seen.add(q)

        return unique[:2]


    def _build_prompt(self, processed_input: dict, domain_tag: str = "general") -> str:
        """
        Builds a strict classification prompt with:
        - hard JSON schema (no 'reason' in output)
        - 2-shot (1 YES + 1 NO) closest in length to input
        - includes human-readable Reason lines under examples
        - final question to classify

        Expects processed_input['BERT_Input'] to hold the normalized question text.
        """
        # i have no idea why i added this import but it works so im leaving it
        import json as _json

        question = processed_input["BERT_Input"]
        qlen = len(question)

        # Get balanced two-shot examples (closest length)
        shots = self._pick_two_balanced_examples(domain_tag, qlen)

        # Only expose the required JSON fields in examples (schema!), but show reason separately
        def example_block(q, meta):
            # keep only schema fields in the example JSON
            ej = {"search_needed": int(meta.get("search_needed", 0)),
                "confidence": float(meta.get("confidence", 0.0))}
            reason = meta.get("reason", "")
            return f"Q: {q}\nA: {_json.dumps(ej, ensure_ascii=False)}\nReason: {reason}"

        examples_str = "\n\n".join(example_block(q, a) for q, a in shots)

        # Build the final instruction
        return f"""
    You are a highly accurate text classifier.

    TASK:
    - Decide if the input question REQUIRES an external web search.
    - 1 means search needed, 0 means not needed. return as integer.
    - Output ONLY valid JSON with fields EXACTLY as specified:
    - "search_needed": 1 or 0
    - "confidence": float between 0 and 0.7

    RULES:
    - DO NOT include any field other than the two specified.
    - Ignore any 'Reason:' lines in the examples; they are for illustration only.
    - KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA. {{"search_needed":0,"confidence":0.2}}
    - Heuristics:
    - 1: needs fresh or volatile info (weather, news, prices, schedules, leadership, “latest”, “current”, release dates).
    - 0: stable facts, definitions, math, basic programming or best-practice guidance you can answer without lookup.
    - Confidence: start at 0.50, adjust up/down in 0.05 or 0.10 increments based on certainty.
    Examples ({domain_tag} domain; closest in length to the input):
    {examples_str}

    Now classify this:
    {question}
    A(PICK THE CORRECT CLASSIFICATION BUT PICK A LOWER CONFIDANCE SCORE):""".strip()

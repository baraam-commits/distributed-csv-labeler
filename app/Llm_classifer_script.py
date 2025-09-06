import ollama
from Prosses_user_input import ProssesUserInput as PUI
import json
import re
from typing import List, Tuple, Dict


# Examples for few-shot prompting, categorized by domain. 
EXAMPLES: Dict[str, List[Tuple[str, dict]]] = {
    "wikipedia": [
        ("where are the rocky mountains located in colorado", {"search_needed": 0, "confidence": 0.9}),
        ("who plays tupac mother in all eyez on me", {"search_needed": 1, "confidence": 0.9}),
        ("where does the atp from cellular respiration end up", {"search_needed": 0, "confidence": 0.92}),
        ("who is the youngest grand master in chess", {"search_needed": 1, "confidence": 0.93}),
    ],
    "programming": [
        ("I'm having trouble understanding and using Django's ImageField", {"search_needed": 0, "confidence": 0.86}),
        ("error: spawn EACCES in gulp pipeline", {"search_needed": 0, "confidence": 0.85}),
        ("how to split a git history based on a target directory", {"search_needed": 1, "confidence": 0.9}),
        ("what is the latest stable pytorch version", {"search_needed": 1, "confidence": 0.95}),
    ],
    "medical": [
        ("what are the treatments for Lymphocytic Choriomeningitis?", {"search_needed": 1, "confidence": 0.93}),
        ("who is at risk for Loiasis?", {"search_needed": 1, "confidence": 0.9}),
        ("what is (are) Parasites - Loiasis?", {"search_needed": 0, "confidence": 0.85}),
        ("how can botulism be prevented?", {"search_needed": 0, "confidence": 0.82}),
    ],
    "mental_health": [
        ("I keep having these random thoughts that I don't want. Things like 'you aren't worth anything.' I know they're my own thoughts but it feels like someone else is saying it. What is wrong with me, and how can I stop having these thoughts?", {"search_needed": 0, "confidence": 0.8}),
        ("My boyfriend is in recovery from drug addiction. We recently got into a fight and he has become very distant. I don't know what to do to fix the relationship.", {"search_needed": 0, "confidence": 0.82}),
        ("What am I doing wrong? My wife and I are fighting all the time. What can I do?", {"search_needed": 0, "confidence": 0.8}),
    ],
    "general": [
        ("who is the ceo of openai", {"search_needed": 1, "confidence": 0.9}),
        ("weather in nyc tomorrow?", {"search_needed": 1, "confidence": 0.96}),
        ("define convolution in signal processing", {"search_needed": 0, "confidence": 0.85}),
        ("rare beauty annual revenue last year", {"search_needed": 1, "confidence": 0.92}),
        ("is the empire state building taller than the shard?", {"search_needed": 0, "confidence": 0.88}),
        ("what are the current us interest rates", {"search_needed": 1, "confidence": 0.95}),
    ],
}


DEFAULT_EXAMPLES = EXAMPLES["general"]
DEFAULT_SYSTEM_PROMPT = """
    You are a highly accurate text classifier.

    TASK:
    - Decide if the input question REQUIRES an external web search.
    - 1 means search needed, 0 means not needed. return as integer.
    - Output ONLY valid JSON with fields EXACTLY as specified:
    - "search_needed": 1 or 0
    - "confidence": float 0.0–1.0


    GUIDELINES:
    - 1: needs fresh info (weather, revenue, leadership, news, prices, schedules).
    - 0: basic facts, definitions, arithmetic.
    - Confidence: 1.0 only if trivial.
    - if you believe you would be able to answer the question with high confidence without a search, return 0. if you believe you would need to look up information to answer the question, return 1.
    - KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA. {"search_needed":0,"confidence":1.0} DO NOT GENERATE EXTRA WHITESPACE.

    EXAMPLES:

    Q: define convolution in signal processing <ENT> entity: Signal Processing, type: ORG </ENT>
    A: {"search_needed":0,"confidence":0.85}

    Q: rare beauty annual revenue last year <ENT> entity: annual, type: DATE; entity: last year, type: DATE </ENT>
    A: {"search_needed":1,"confidence":0.92}

    Q: what is 2 + 2? <ENT> </ENT>
    A: {"search_needed":0,"confidence":1.0}
    """

class llmClassifier:
    def __init__(self, model: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT, options: dict = None):
        self.model = model
        self.system_prompt = system_prompt
        self.options = options or {}
        self.client = ollama
        self.pui = PUI()
        self.retry_attempts = 0  # Number of retry attempts for model query
        
        # Ensure the model is installed
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


    def classify(self, user_input: str, domain_tag: str = "general"):
        
        processed_input = self.pui.process_user_input(user_input)

        if len(processed_input["tokenized"]) == 0:
            # No valid input provided
            return {"search_needed": 0, "confidence": 0.0}
        elif len(processed_input["tokenized"]) > 512:
            # Input too long for model reasoning -1 indicates too long
            return {"search_needed": 0, "confidence": 0.0}

        prompt = self._build_prompt(processed_input, domain_tag)

        if self.options == None:
                response = self.client.generate(model=self.model, prompt=prompt, system=self.system_prompt)
        else:
            response = self.client.generate(model=self.model, prompt=prompt, system=self.system_prompt, options=self.options)

        try:
            if response["response"] is None:
                raise TypeError("No response from model")

            response_data = self._load_json_or_raise(response["response"])

            if "search_needed" not in response_data:
                raise KeyError("search_needed")
            if "confidence" not in response_data:
                raise KeyError("confidence")

            
            self.retry_attempts = 0  # Reset retry attempts on success
            return response_data
        except (TypeError, KeyError) as e:
            print(f"Error processing response for question '{processed_input['BERT_Input']}': {e}")
            print(f"Model output: {response}")
            self.retry_attempts += 1
            if self.retry_attempts < 3:
                print(f"Retrying... Attempt {self.retry_attempts}")
                return self.classify(user_input)
            else:
                print("Max retry attempts reached. Returning default response.")
                self.retry_attempts = 0
                # Standardized fallback response reasoning -2 indicated unable to classify
                return {"search_needed": 0, "confidence": 0.0}
            
        except Exception as e:
            print(f"Unexpected error for '{processed_input['BERT_Input']}': {e}")
            print(f"Model output: {response["response"]}\n")

            self.retry_attempts += 1
            if self.retry_attempts < 3:
                print(f"Retrying... Attempt {self.retry_attempts}")
                return self.classify(user_input)
            else:
                print("Max retry attempts reached. Returning default response.")
                self.retry_attempts = 0
                # Standardized fallback response reasoning -2 indicated unable to classify
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


    def _build_prompt(self, processed_input: dict, domain_tag: str = "general") -> str:
        """
        Builds a strict classification prompt with:
        - hard JSON schema (no reasoning)
        - 2 YES + 2 NO few-shots closest in length to input
        - final question to classify
        Expects processed_input['BERT_Input'] to hold the normalized question text.
        """
        question = processed_input["BERT_Input"]
        qlen = len(question)

        shots = self._closest_examples(domain_tag.lower(), target_len=qlen, need_yes=2, need_no=2)
        examples_str = "\n\n".join([f"Q: {q}\nA: {json.dumps(a, ensure_ascii=False)}" for q, a in shots])

        return f"""
    You are a highly accurate text classifier.
    Your task: decide if the input question REQUIRES an external web search based on your system prompt.
    1 for search, 0 for no search.
    Always output ONLY valid JSON with these fields exactly as specified with the same names and values:
    - "search_needed": 1 or 0
    - "confidence": float 0.0–1.0

    KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA.
    {{"search_needed":0,"confidence":1.0}}

    Examples ({domain_tag} domain; closest in length to the input):
    {examples_str}

    Now classify this:
    {question}
    A:
    """.strip()
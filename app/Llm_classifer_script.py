import ollama
from Prosses_user_input import ProssesUserInput as PUI
import json
import re
from collections import OrderedDict
class llmClassifier:
    def __init__(self, model: str, system_prompt: str = "", options: dict = None):
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


    def classify(self, user_input: str):
        
        processed_input = self.pui.process_user_input(user_input)

        if len(processed_input["tokenized"]) == 0:
            # No valid input provided
            return {"search_needed": 0, "confidence": 0.0}
        elif len(processed_input["tokenized"]) > 512:
            # Input too long for model reasoning -1 indicates too long
            return {"search_needed": 0, "confidence": 0.0}

        prompt = f"""
        You are a highly accurate text classifier.
        Your task: decide if the input question REQUIRES an external web search based on your system prompt.
        1 for search 0 for no search Always output ONLY valid JSON with these fields exactly as specified
        with the same names and values:
        - Output ONLY valid JSON with fields EXACTLY as specified:
        - \"search_needed\": 1 or 0
        - \"confidence\": float 0.0–1.0
        KEEP FORMAT CONSISTENT. ONE line ONLY. THIS EXACT SCHEMA. {{\"search_needed\":0,\"confidence\":1.0}}
        keeping that in mind classify this {processed_input['BERT_Input']}
        """
        
        response = self.client.generate(model=self.model, prompt=prompt, system=self.system_prompt, options=self.options)

        try:
            if response["response"] is None:
                raise TypeError("No response from model")

            response_data = self.load_json_or_raise(response["response"])

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


    def load_json_or_raise(self, text: str):
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

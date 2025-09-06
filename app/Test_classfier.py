import random
import time
import ollama
from Llm_classifer_script import llmClassifier as classifier
import os
import json
import csv



# prechecked classifications for accuracy verification related to index of questions list above
labeled_questions = {
    "general": {
        "character sketch of charlie from charlie and the chocolate factory": {"search_needed": 0, "confidence": 0.82},
        "night of the chicken dead full movie free download": {"search_needed": 1, "confidence": 0.9},
        "mrs frisby and the rats of nimh ending": {"search_needed": 0, "confidence": 0.78},
        "where are they building the new raider stadium": {"search_needed": 1, "confidence": 0.92},
        "what do you get for winning the crossfit games": {"search_needed": 1, "confidence": 0.9},
        "who is 30 seconds to mars touring with": {"search_needed": 1, "confidence": 0.88},
        "what is the origin of the shih tzu": {"search_needed": 0, "confidence": 0.8},
        "when was big brothers big sisters canada founded": {"search_needed": 1, "confidence": 0.9},
        "how is the flag draped over a casket": {"search_needed": 1, "confidence": 0.86},
        "where does the name pg tips come from": {"search_needed": 1, "confidence": 0.86},
        "can minors drink with their parents in wisconsin": {"search_needed": 1, "confidence": 0.93},
        "where were the original planet of the apes filmed": {"search_needed": 1, "confidence": 0.88},
        "how many episodes in season 5 sex and the city": {"search_needed": 1, "confidence": 0.9},
        "who is the youngest grand master in chess": {"search_needed": 1, "confidence": 0.94},
        "where is the source and mouth of the mississippi river located": {"search_needed": 1, "confidence": 0.9},
    },

    "mental_health": {
        "I get so depressed  because of my dad's yelling. He keeps asking me why I can't just be happy the way I am and yells at me on a daily basis. Is this considered emotional abuse?": {"search_needed": 0, "confidence": 0.86},
        "I have a friend that who I used to be in a relationship with. It was brief and turned into us being just good friends.\n\nI spent the weekend with him and  it upset my boyfriend. Was i wrong?": {"search_needed": 0, "confidence": 0.84},
        "I crossdress and like to be feminine but I am attracted to women, but yet that seems to bother girls I date or ask out.\n\nHow can I approach them about it? should I hold back and keep it a secret, or should I just be up-front about it.  I wonder if i should stop or if I should continue to do it since it makes me happy.  What should I do?": {"search_needed": 0, "confidence": 0.86},
        "I don't know how else to explain it. All I can say is that I feel empty, I feel nothing.  How do I stop feeling this way?": {"search_needed": 0, "confidence": 0.86},
        "I'm dealing with an illness that will never go away and I feel like my life will never change for the better. I feel alone and that i have no one.\n\nHow can I overcome this pain and learn to be happy alone?": {"search_needed": 0, "confidence": 0.86},
        "I am in my early 20s and I still live with my parents because I can't afford to live alone.\n\nMy mother says that if I live under her roof I have to follow her rules. She is trying to control my life. What should I do?": {"search_needed": 0, "confidence": 0.84},
        "I'm concerned about My 12 year old daughter.\n\nAbout a month or two ago she started walking on her toes, as well as coloring and writing very messy. This all happened very suddenly. She has never walked on her tiptoes and has always colored and written very neatly.\n\nIs this something I should be concerned abou? Any advice will help.": {"search_needed": 0, "confidence": 0.7},
        "A few years ago I was making love to my wife when for no known reason I lost my erection,\n\nNow I'm In my early 30s and my problem has become more and more frequent.  This is causing major problems for my ego and it's diminishing my self esteem. This has resulted in ongoing depression and tearing apart my marriage.\n\nI am devastated and cannot find a cause for these issues. I am very attracted to my wife and want to express it in the bedroom like I used to.\n\nWhat could be causing this, and what can I do about it?": {"search_needed": 0, "confidence": 0.78},
        "I've been bullied for years and the teachers have done nothing about it. I haven't been diagnosed with depression, but i have been extremely sad for years.\n\nHow can I deal with being bullied at school when the teachers won't help?": {"search_needed": 0, "confidence": 0.84},
        "I'm dealing with imposter syndrome in graduate school.  I know that by all accounts I am a phenomenal graduate student, and that I am well-published.  I am well liked by students and faculty alike.  And yet I cannot shake the feeling that I'm going to be found out as a fraud.\n\nHow can I get over this feeling?": {"search_needed": 0, "confidence": 0.86},
        "I'm in my late teens and live with my dad.  The only time I go out is for my college classes. Sometimes when I see my friends I want to talk with them, but sometimes I won't want to talk to them for days or even weeks.\n\nSometimes I feel i'm not worth knowing or i'm never going to do anything right.": {"search_needed": 0, "confidence": 0.84},
        "I have social anxiety and avoid group hangouts even with friends. How can I start pushing myself without panicking?": {"search_needed": 0, "confidence": 0.82},
        "My mind races at night and I can't sleep even when I'm exhausted. Any practical steps to calm down before bed?": {"search_needed": 0, "confidence": 0.82},
        "How do I set boundaries with a controlling parent without starting constant fights?": {"search_needed": 0, "confidence": 0.83},
        "How can I support my partner who is struggling with depression without burning out myself?": {"search_needed": 0, "confidence": 0.84},
    },

    "programming": {
        "I am having 4 different tables like select * from System select * from Set select * from Item select * from Versions Now for each system Id there will be n no.of Sets, and foe each set there qill be n no. of Items and for each item there will be n no. of Versions. each system has n no of set each Set has n no of Items each Item has n no of Versions So, Now when i give SystemId then i have to retrieve all the records from Set and Items of each set and Versions of each Items in single storedprocedure.": {"search_needed": 0, "confidence": 0.8},
        "I have two table m_master and tbl_appointment [This is tbl_appointment table][1] [This is m_master table][2]": {"search_needed": 0, "confidence": 0.76},
        "I'm trying to extract US states from wiki URL, and for which I'm using Python Pandas. However, the above code is giving me an error ... installed html5lib and beautifulsoup4 as well, but it is not working.": {"search_needed": 0, "confidence": 0.78},
        "I'm so new to C#, I wanna make an application that can easily connect to the SqlServer database... my reader always gives Null": {"search_needed": 0, "confidence": 0.82},
        "basically i have this array ... if an element['sub'] appears twice ... both instances should be next to each other in the array (PHP)": {"search_needed": 0, "confidence": 0.84},
        "I am trying to make a constructor for a derived class. Error: no default constructor exists for class 'FirstClass'": {"search_needed": 0, "confidence": 0.88},
        "I am using c++ ... create an array that may be change in dimensions ... double X[I][J];": {"search_needed": 0, "confidence": 0.88},
        "I'm getting a bit lost in TS re-exports ... What's the right way to do a rollup like this?": {"search_needed": 0, "confidence": 0.76},
        "I am trying out the new Fetch API but having trouble with Cookies ... Fetch seems to ignore Cookie header": {"search_needed": 0, "confidence": 0.8},
        "How can I proceed to print the list content like this ?": {"search_needed": 0, "confidence": 0.7},
        "Written the below code trying to identify all primes up 100 ... why doesn't it work?": {"search_needed": 0, "confidence": 0.9},
        "app/boot.ts app/app.component.ts Error:": {"search_needed": 1, "confidence": 0.7},
        "We cannot alter the HTML; two submit buttons have the same id. How to isolate each for different onclick?": {"search_needed": 0, "confidence": 0.86},
        "Run a process in a thread that times out after 30s; use join/is_alive and Event. Is this pythonic?": {"search_needed": 0, "confidence": 0.88},
        "Error: spawn EACCES in gulp-imagemin/exec-buffer pipeline": {"search_needed": 0, "confidence": 0.86},
    },

    "math": {
        "machine a produces 100 parts twice as fast as machine b does . machine b produces 100 parts in 60 minutes . if each machine produces parts at a constant rate , how many parts does machine a produce in 6 minutes ?": {"search_needed": 0, "confidence": 0.98},
        "if the area of a square with sides of length 3 centimeters is equal to the area of a rectangle with a width of 4 centimeters , what is the length of the rectangle , in centimeters ?": {"search_needed": 0, "confidence": 0.98},
        "if n is a prime number greater than 5 , what is the remainder when n ^ 2 is divided by 12 ?": {"search_needed": 0, "confidence": 0.99},
        "set j consists of 5 consecutive even numbers . if the smallest term in the set is - 2 , what is the range of the positive integers in set j ?": {"search_needed": 0, "confidence": 0.97},
        "what is the greatest positive integer n such that 3 ^ n is a factor of 36 ^ 450 ?": {"search_needed": 0, "confidence": 0.98},
        "the sum of all the integers g such that - 26 < g < 24 is": {"search_needed": 0, "confidence": 0.98},
        "a sum of money at simple interest amounts to $ 680 in 3 years and to $ 710 in 4 years . the sum is :": {"search_needed": 0, "confidence": 0.98},
        "a student chose a number , multiplied it by 2 , then subtracted 180 from the result and got 104 . what was the number he chose ?": {"search_needed": 0, "confidence": 0.99},
        "two brothers take the same route to school on their bicycles , one gets to school in 25 minutes and the second one gets to school in 36 minutes . the ratio of their speeds is": {"search_needed": 0, "confidence": 0.98},
        "the pinedale bus line travels at an average speed of 60 km / h , and has stops every 5 minutes along its route . yahya wants to go from his house to the pinedale mall , which is 9 stops away . how far away , in kilometers , is pinedale mall away from yahya ' s house ?": {"search_needed": 0, "confidence": 0.96},
        "in a certain warehouse , 50 percent of the packages weigh less than 75 pounds , and a total of 48 packages weigh less than 25 pounds . if 80 percent of the packages weigh at least 25 pounds , how many of the packages weigh at least 25 pounds but less than 75 pounds ?": {"search_needed": 0, "confidence": 0.97},
        "in one hour , a boat goes 11 km along the stream and 5 km against the stream . the speed of the boat in still water ( in km / hr ) is :": {"search_needed": 0, "confidence": 0.98},
        "the ratio of the cost price and the selling price is 4 : 5 . the profit percent is ?": {"search_needed": 0, "confidence": 0.99},
        "if 45 % of a class averages 100 % on a test , 50 % of the class averages 78 % on the test , and the remainder of the class averages 65 % on the test , what is the overall class average ? ( round final answer to the nearest percent ) .": {"search_needed": 0, "confidence": 0.99},
        "of the votes cast on a certain proposal , 62 more were in favor of the proposal than were against it . if the number of votes against the proposal was 40 percent of the total vote , what was the total number of votes cast ? ( each vote cast was either in favor of the proposal or against it . )": {"search_needed": 0, "confidence": 0.98},
    }
}

CSV_HEADERS = [
    "timestamp",
    "model",
    "total_questions",
    "cpu_time_total",
    "cpu_time_avg",
    "latency_total",
    "latency_avg",
    "search_count",
    "no_search_count",
    "avg_confidence",
    "errors",
    "discrepancies_total",
    "avg_delta_conf",
    "avg_abs_delta_conf",
    "options_json",
    "domains_json",
]

def _flatten_labeled_data(labeled):
    items = []
    is_nested = all(
        isinstance(v, dict) and (not v or isinstance(next(iter(v.values())), dict))
        for v in labeled.values()
    )
    if is_nested:
        for domain, qmap in labeled.items():
            for q, lbl in qmap.items():
                items.append((domain, q, lbl))
    else:
        for q, lbl in labeled.items():
            items.append(("default", q, lbl))
    return items

def _test_model(model, system_prompt, options, labeled_data):
    print("\n\n\n\n")
    classifying_model = classifier(model, system_prompt, options)

    print(f"--- Testing {model} ---\n")

    dataset = _flatten_labeled_data(labeled_data)
    total_q = len(dataset)

    model_times = []
    start_cpu_time = time.process_time()

    no_search_count = 0
    search_count = 0
    avg_confidence = 0.0
    errors = 0

    per_domain = {}
    # per_domain[domain] = {
    #   "results": {question: result_dict},
    #   "search_count": int,
    #   "no_search_count": int,
    #   "avg_conf": float_accum,
    #   "errors": int,
    #   "delta_confs": [float, ...],   # result_conf - gold_conf for discrepant items
    # }

    for domain, question, gold in dataset:
        if domain not in per_domain:
            per_domain[domain] = {
                "results": {},
                "search_count": 0,
                "no_search_count": 0,
                "avg_conf": 0.0,
                "errors": 0,
                "delta_confs": [],
            }

        loop_start = time.perf_counter()
        result = classifying_model.classify(question)
        loop_end = time.perf_counter()
        model_times.append(loop_end - loop_start)

        if result is None:
            print("\n\n\n\n")
            print(f"Model returned None for '{question}'")
            print(f"Model output: {result}\n")
            print(f"ideal result: {gold}\n")
            print("running classify again to see if it's consistent\n\n")
            print(classifying_model.classify(question))
            print("Exiting test")
            print("\n\n\n\n")
            return None  # keep your early exit behavior

        try:
            if result["search_needed"] == 1:
                search_count += 1
                per_domain[domain]["search_count"] += 1
            else:
                no_search_count += 1
                per_domain[domain]["no_search_count"] += 1

            avg_confidence += result["confidence"]
            per_domain[domain]["avg_conf"] += result["confidence"]
            per_domain[domain]["results"][question] = result

        except TypeError as t:
            print(f"Model returned incorrect json key or value for '{question}': {t}")
            print(f"Model output: {result}")
            errors += 1
            per_domain[domain]["errors"] += 1
        except KeyError as k:
            print(f"Model returned incorrect json key or value for '{question}': {k}")
            print(f"Model output: {result}")
            errors += 1
            per_domain[domain]["errors"] += 1
        except Exception as e:
            print(f"Unexpected error for '{question}': {e}")
            print(f"Model output: {result}")

    end_cpu_time = time.process_time()

    cpu_time = end_cpu_time - start_cpu_time
    total_time = sum(model_times)
    avg_conf_overall = (avg_confidence / total_q) if total_q else 0.0

    print(f"--- {model} Performance Metrics ---")
    print(f"CPU time for {total_q} questions: {cpu_time:.2f} s")
    print(f"Total generation time for {total_q} questions: {total_time:.2f} s")
    print()
    print(f"Average CPU time per question: {cpu_time/total_q:.4f} s")
    print(f"Average generation time per question: {total_time/len(model_times):.4f} s")
    print()
    print(f"Questions needing search: {search_count} ({(search_count/total_q)*100:.2f}%)")
    print(f"Questions NOT needing search: {no_search_count} ({(no_search_count/total_q)*100:.2f}%)")
    print(f"Average confidence: {avg_conf_overall:.3f}")
    print()
    print(f"Errors: {errors} ({(errors/total_q)*100:.2f}%)")
    print("\n")

    # Domain-specific discrepancies and deltas
    print("=== Domain-specific Discrepancies ===")
    grand_discrepancies = 0
    grand_deltas = []

    # Build gold lookup
    gold_map = {}
    if any(isinstance(v, dict) and (not v or isinstance(next(iter(v.values())), dict)) for v in labeled_data.values()):
        nested = labeled_data
    else:
        nested = {"default": labeled_data}
    for d, qmap in nested.items():
        for q, g in qmap.items():
            gold_map[(d, q)] = g

    domain_metrics = {}
    for domain, bucket in per_domain.items():
        results = bucket["results"]
        if not results:
            continue

        dom_total = len(results)
        dom_avg_conf = bucket["avg_conf"] / dom_total if dom_total else 0.0
        dom_search = bucket["search_count"]
        dom_no_search = bucket["no_search_count"]

        dom_disc = 0
        dom_deltas = []

        for q, res in results.items():
            gold = gold_map.get((domain, q)) or gold_map.get(("default", q))
            if not gold:
                continue
            if res.get("search_needed") != gold.get("search_needed"):
                gconf = gold.get("confidence", 0.0)
                rconf = res.get("confidence", 0.0)
                delta = rconf - gconf
                dom_deltas.append(delta)
                print(f"[{domain}] Discrepancy: '{q[:80] + ('...' if len(q)>80 else '')}'")
                print(f"  Δ confidence (result - label): {delta:+.2f}")
                print(f"  Expected: {gold['search_needed']}  |  Got: {res['search_needed']}")
                dom_disc += 1

        grand_discrepancies += dom_disc
        grand_deltas.extend(dom_deltas)

        print(f"[{domain}] totals: {dom_disc} discrepancies out of {dom_total} "
              f"({(dom_disc/dom_total)*100:.2f}%) | "
              f"search={dom_search}, no_search={dom_no_search}, avg_conf={dom_avg_conf:.3f}")

        domain_metrics[domain] = {
            "domain": domain,
            "total": dom_total,
            "search": dom_search,
            "no_search": dom_no_search,
            "avg_confidence": dom_avg_conf,
            "discrepancies": dom_disc,
            "discrepancies_pct": (dom_disc/dom_total) if dom_total else 0.0,
            "avg_delta_conf": (sum(dom_deltas)/len(dom_deltas)) if dom_deltas else 0.0,
        }

    overall_avg_delta = (sum(grand_deltas)/len(grand_deltas)) if grand_deltas else 0.0
    print(f"\nOverall discrepancies: {grand_discrepancies} out of {len(dataset)} "
          f"({(grand_discrepancies/len(dataset))*100:.2f}%)")

    # Optional: unload
    ollama.generate(model=model, prompt='', keep_alive=0)

    # ---- return structured metrics for CSV logging ----
    overall_avg_delta = (sum(grand_deltas)/len(grand_deltas)) if grand_deltas else 0.0
    overall_avg_abs_delta = (sum(abs(x) for x in grand_deltas)/len(grand_deltas)) if grand_deltas else 0.0
    now_ts = time.time()

    return {
        "timestamp": now_ts,
        "model": model,
        "options": dict(options) if options else {},
        "total_questions": total_q,
        "cpu_time_total": cpu_time,
        "gen_time_total": total_time,
        "cpu_time_avg": cpu_time/total_q if total_q else 0.0,
        "gen_time_avg": total_time/total_q if total_q else 0.0,
        "search_count": search_count,
        "no_search_count": no_search_count,
        "avg_confidence": avg_conf_overall,
        "errors": errors,
        "discrepancies_total": grand_discrepancies,
        "discrepancies_pct": (grand_discrepancies/total_q) if total_q else 0.0,
        "avg_delta_conf_overall": overall_avg_delta,
        "avg_abs_delta_conf_overall": overall_avg_abs_delta,
        "domains": domain_metrics,
    }


def _append_metrics_csv(csv_path: str, metrics: dict):
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    row = {
        "timestamp": metrics.get("timestamp"),
        "model": metrics.get("model"),
        "total_questions": metrics.get("total_questions"),
        "cpu_time_total": f"{metrics.get('cpu_time_total', 0.0):.6f}",
        "cpu_time_avg": f"{metrics.get('cpu_time_avg', 0.0):.6f}",
        # map gen_time_* to latency_* in CSV
        "latency_total": f"{metrics.get('gen_time_total', 0.0):.6f}",
        "latency_avg": f"{metrics.get('gen_time_avg', 0.0):.6f}",
        "search_count": metrics.get("search_count"),
        "no_search_count": metrics.get("no_search_count"),
        "avg_confidence": f"{metrics.get('avg_confidence', 0.0):.6f}",
        "errors": metrics.get("errors"),
        "discrepancies_total": metrics.get("discrepancies_total"),
        # overall deltas mapped to CSV column names
        "avg_delta_conf": f"{metrics.get('avg_delta_conf_overall', 0.0):.6f}",
        "avg_abs_delta_conf": f"{metrics.get('avg_abs_delta_conf_overall', 0.0):.6f}",
        "options_json": json.dumps(metrics.get("options", {}), ensure_ascii=False),
        "domains_json": json.dumps(metrics.get("domains", {}), ensure_ascii=False),
    }

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def test_qwen(system_prompt=None, options=None):
    model = "qwen2.5:0.5b-instruct"
    return _test_model(model, system_prompt, options, labeled_questions)

def test_granite32b(system_prompt=None, options=None):
    model = "granite3.3:2b"
    return _test_model(model, system_prompt, options, labeled_questions)

def test_llama3(system_prompt=None, options=None):
    model = "llama3.2:1b"
    return _test_model(model, system_prompt, options, labeled_questions)

def test_granite31_moe1b(system_prompt=None, options=None):
    model = "granite3.1-moe:1b"
    return _test_model(model, system_prompt, options, labeled_questions)

def test_falcon3_1b(system_prompt=None, options=None):
    model = "falcon3:1b"
    
    return _test_model(model, system_prompt, options, labeled_questions)

def test_phi4_mini_3_8b(system_prompt=None, options=None):
    model = "phi4-mini:3.8b"
    
    ollama.generate(model=model, prompt='')
    return _test_model(model, system_prompt, options, labeled_questions)
# Run tests

# test_granite32b(options=options)
# test_qwen(options=options)
# test_llama3(options=options)
# test_qwen3(options=options)
# test_granite31_moe1b(options=options)
# test_falcon3_1b(options=options)
# test_phi4_mini_3_8b(options=options)

"""
CSV Output Format for Model Testing
-----------------------------------

Each row in results.csv corresponds to one model run with one parameter set.

Columns:
    model              : The Ollama model name (e.g. "phi4-mini-reasoning:3.8b")
    options            : JSON string of decoding options (temperature, top_p, etc.)
    latency_avg        : Average generation latency per question (seconds)
    latency_total      : Total latency across all questions (seconds)
    cpu_time_avg       : Average CPU time per question (seconds)
    cpu_time_total     : Total CPU time (seconds)
    discrepancies_total: Number of mismatches between model output and ground truth
    avg_confidence     : Mean confidence score assigned by the model
    errors             : Count of parsing/JSON errors in this run
    search_count       : Number of questions classified as search_needed=1
    no_search_count    : Number of questions classified as search_needed=0
    total_questions    : Total number of questions evaluated
    domain             : (Optional) Domain tested (e.g. general, medical, programming)

Usage in Excel:
---------------
1. Open results.csv in Excel or Google Sheets.
2. Create a PivotTable:
    - Rows: model
    - Values: latency_avg, discrepancies_total, avg_confidence
3. Insert charts:
    - Bar chart for discrepancies per model
    - Line chart for latency comparison
    - Scatter plot for confidence vs discrepancies

This makes it easy to visually compare models across accuracy, latency,
and confidence without needing any extra code.
"""
def run_and_log_all_tests(csv_path="results.csv", system_prompt=None):
    
    random.seed(42)  # for reproducibility
    base_options = {
        "format": "json",
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_thread": 6,
        "num_predict": 22
    }

    model_fns = [
        test_phi4_mini_3_8b,
        test_granite31_moe1b,
        test_falcon3_1b,
        test_granite32b,
        test_qwen,
        test_llama3,    
    ]

    for i in range(10):
        # Copy base and mutate per run
        options = dict(base_options)

        if i < 5:
            # conservative
            options["temperature"] = round(random.uniform(0.0, 0.3), 2)
            options["top_p"] = round(random.uniform(0.5, 1.0), 2)
            options["top_k"] = random.choice([1, 2, 3, 5])
        else:
            # liberal
            options["temperature"] = round(random.uniform(0.4, 1.0), 2)
            options["top_p"] = round(random.uniform(0.1, 0.5), 2)
            options["top_k"] = int(random.uniform(10, 60))

        # You can also randomize num_predict etc. if you want:
        # options["num_predict"] = int(random.uniform(16, 128))

        print(f"\n==== Sweep {i+1}/10 | options={options} ====\n")
        for fn in model_fns:
            try:
                metrics = fn(system_prompt=system_prompt, options=options)
                _append_metrics_csv(csv_path, metrics)
            except Exception as e:
                # Don’t kill the whole sweep if one model borks
                print(f"[WARN] {fn.__name__} failed: {e}")


run_and_log_all_tests(csv_path="results2.csv")

"""
results.csv will contain all the results for easy analysis in Excel or Sheets. for current method and 4 shot examples in LLM_Classifier.py
results2.csv will contain results for 2-shot with reasoning for classification
"""

# legacy test function kept for reference
#     start_cpu_time = time.process_time()
#     no_search_count = 0
#     search_count = 0
#     avg_confidence = 0.0
#     errors = 0
#     count = 0
#     results = {}
#     # main loop ment to be used for profiling
#     for question in questions:

#         # if count > 10: # for quicker testing
#         #     break
#         # count += 1
        
#         loop_start = time.perf_counter()
#         result = classifying_model.classify(question)
#         loop_end = time.perf_counter()
#         model_times.append(loop_end - loop_start)
        
#         if result is None:
#             print("\n\n\n\n")
#             print(f"Model returned None for '{question}'")


#             print(f"Model output: {result}\n")
#             print(f"ideal result: {labeled_questions[question]}\n")
#             print("running classify again to see if it's consistent\n\n")
#             print(classifying_model.classify(question))
            
#             print("Exiting test")
#             print("\n\n\n\n")

#             return
#         try:
#             if result["search_needed"] == 1:
#                 search_count += 1
#             else:
#                 no_search_count += 1
#             avg_confidence += result["confidence"]
#             results[question] = result 
#         except TypeError as t:
#             print(f"Model returned incorrect json key or value for '{question}': {t}")
#             print(f"Model output: {result}")
#             errors += 1
#         except KeyError as k:
#             print(f"Model returned incorrect json key or value for '{question}': {k}")
#             print(f"Model output: {result}")
#             errors += 1
#         except Exception as e:
#             print(f"Unexpected error for '{question}': {e}")
#             print(f"Model output: {result}")
            
#     end_cpu_time = time.process_time()
  
    
#     cpu_time = end_cpu_time - start_cpu_time
#     total_time = sum(model_times)

#     print ("--- " + model + "'s" + " Performance Metrics ---")
    
#     print (f"Cpu time for {len(questions)} questions: {cpu_time:.2f} seconds")
#     print (f"Total time for {len(questions)} questions: {total_time:.2f} seconds")
#     print()
#     print (f"Average cpu time per question: {cpu_time/len(questions):.2f} seconds")
#     print (f"Average time per question: {total_time/len(model_times):.2f} seconds")
#     print()
#     print (f"Questions needing search: {search_count} ({(search_count/len(questions))*100:.2f}%)")
#     print (f"Questions NOT needing search: {no_search_count} ({(no_search_count/len(questions))*100:.2f}%)")
#     print (f"Average confidence: {(avg_confidence/len(questions)):.2f}")
#     print()
#     print (f"Errors: {errors} ({(errors/len(questions))*100:.2f}%)")
#     print("\n\n\n\n\n")
    
#     # accuracy verification
#     result_discrepancies = 0
#     for question, result in results.items():
#         if result["search_needed"] != labeled_questions[question]["search_needed"]:
#             print(f"Discrepancy for question '{question}':")
#             print(f"  Difference in Confidence (result - label):  {result['confidence'] - labeled_questions[question]['confidence']:.2f}")
#             print(f"  Expected: {labeled_questions[question]['search_needed']}")
#             print(f"  Got:      {result['search_needed']}")
#             result_discrepancies += 1
        
#     print(f"Total discrepancies: {result_discrepancies} out of {len(results)} ({(result_discrepancies/len(results))*100:.2f}%)")
#     ollama.generate(model=model, prompt='', keep_alive=0)
# """"""
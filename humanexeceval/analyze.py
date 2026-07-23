"""HumanExecEval pipeline, step 1/2.

Builds the full per-task difficulty table over all 1745 published functional-MATH tasks
by combining, for each `subject/id`:
  - the 3 monthly snapshot instances (Oct/Nov/Dec-2023): MATH level, type, per-seed answer;
  - empirical difficulty from intermediate/correct_<model>-<snapshot>/ (12 usable models).
Writes `full_tasks.jsonl` next to this script; step 2 (build_spectrum.py) consumes it.

Run:  python3 humanexeceval/analyze.py
"""
import json, os, re, glob
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # repo root (contains the snapshots + intermediate/)
SNAPS = ["Oct-2023", "Nov-2023", "Dec-2023"]
STATIC_SENTINEL = "Jan-1984"                        # sentinel snapshot date == the static MATH benchmark

# Models with populated correct_ dirs (gemini excluded: dirs are empty). Strongest -> weakest-ish.
MODELS = [
    "gpt-4", "gpt-3.5-turbo", "claude-2.1", "mistral-medium",
    "mistralai_Mixtral-8x7B-v0.1", "mistralai_Mixtral-8x7B-Instruct-v0.1",
    "togethercomputer_llama-2-70b", "zero-one-ai_Yi-34B", "zero-one-ai_Yi-34B-Chat",
    "WizardLM_WizardCoder-Python-34B-V1.0",
    "togethercomputer_StripedHyena-Hessian-7B", "togethercomputer_StripedHyena-Nous-7B",
]

SUBJ_DIRS = ["algebra","counting_and_probability","geometry","intermediate_algebra",
             "number_theory","prealgebra","precalculus"]

# ---- 1. Load canonical problem set from snapshots (all 3 seeds) ----
# key = (subject, id)
problems = {}
for subj in SUBJ_DIRS:
    for fp in glob.glob(f"{ROOT}/{SNAPS[0]}/test/{subj}/*.json"):
        pid = os.path.splitext(os.path.basename(fp))[0]
        problems[(subj, pid)] = {"subject": subj, "id": pid}

def load_json(fp):
    with open(fp) as f: return json.load(f)

# gather snapshot data: level, type, per-seed solution, per-seed problem text
def extract_boxed(sol):
    # snapshot solutions are like \boxed{...}
    m = re.search(r"\\boxed\{(.*)\}", sol, re.DOTALL)
    return m.group(1).strip() if m else sol.strip()

for (subj, pid), rec in problems.items():
    seeds_ans, plens = [], []
    level = None; typ = None
    for s in SNAPS:
        fp = f"{ROOT}/{s}/test/{subj}/{pid}.json"
        d = load_json(fp)
        level = d.get("level"); typ = d.get("type")
        seeds_ans.append(extract_boxed(d.get("solution","")))
        plens.append(len(d.get("problem","")))
    rec["level"] = int(level.replace("Level ","")) if level and "Level" in level else None
    rec["type"] = typ
    rec["seed_answers"] = seeds_ans
    rec["n_distinct_answers"] = len(set(seeds_ans))
    rec["max_answer_len"] = max(len(a) for a in seeds_ans)
    rec["avg_problem_len"] = round(sum(plens)/len(plens))
    # answer type heuristic
    a0 = seeds_ans[0]
    if re.fullmatch(r"-?\d+", a0): atype="int"
    elif re.search(r"\\frac|/", a0): atype="fraction"
    elif re.search(r"\\begin|\\pmatrix|,", a0): atype="tuple/matrix/list"
    elif re.search(r"[a-zA-Z\\]", a0): atype="expression"
    else: atype="other"
    rec["answer_type"] = atype

# ---- 2. Per-model correctness (functional per-snapshot + static) ----
def correct_set(model, snap):
    base = f"{ROOT}/intermediate/correct_{model}-{snap}"
    s = set()
    if not os.path.isdir(base): return s
    for subj in SUBJ_DIRS:
        for fp in glob.glob(f"{base}/{subj}/*.txt"):
            pid = os.path.splitext(os.path.basename(fp))[0]
            s.add((subj, pid))
    return s

model_fn = {}      # model -> {key: n_snapshots_solved 0..3}
model_static = {}  # model -> set of keys solved static
for m in MODELS:
    per = defaultdict(int)
    for s in SNAPS:
        for k in correct_set(m, s):
            per[k]+=1
    model_fn[m]=per
    model_static[m]=correct_set(m, STATIC_SENTINEL)

# ---- 3. Aggregate difficulty signals ----
for k, rec in problems.items():
    gpt4_fn = model_fn["gpt-4"].get(k,0)
    gpt4_static = 1 if k in model_static["gpt-4"] else 0
    ens_fn_snapshots = sum(model_fn[m].get(k,0) for m in MODELS)          # 0..36
    n_models_fn_any = sum(1 for m in MODELS if model_fn[m].get(k,0)>0)     # 0..12
    n_models_fn_robust = sum(1 for m in MODELS if model_fn[m].get(k,0)==3) # solved all 3
    n_models_static = sum(1 for m in MODELS if k in model_static[m])       # 0..12
    rec.update(dict(gpt4_fn=gpt4_fn, gpt4_static=gpt4_static,
                    ens_fn_snapshots=ens_fn_snapshots,
                    n_models_fn_any=n_models_fn_any,
                    n_models_fn_robust=n_models_fn_robust,
                    n_models_static=n_models_static))
    # empirical solvability in [0,1]
    S_gpt4 = gpt4_fn/3.0
    S_ens = n_models_fn_any/len(MODELS)
    solvability = 0.6*S_gpt4 + 0.4*S_ens
    rec["solvability"] = round(solvability,4)
    rec["reasoning_gap_flag"] = (gpt4_static==1 and gpt4_fn<3)  # solved static, shaky functional

# ---- 4. Difficulty tier (T1 easiest .. T5 hardest) ----
def tier(rec):
    g=rec["gpt4_fn"]; any_m=rec["n_models_fn_any"]; ens=rec["ens_fn_snapshots"]
    if g==3 and any_m>=4: return 1          # reliably solved by gpt4 + broad
    if g>=2: return 2                        # gpt4 mostly solves
    if g==1 or any_m>=2: return 3            # flaky / only weak solvers
    if ens>0: return 4                       # solved by exactly one model, <3 snaps
    return 5                                 # unsolved functionally by ALL models
for rec in problems.values():
    rec["tier"] = tier(rec)

# ---- 5. Emit full table ----
out_path = os.path.join(HERE, "full_tasks.jsonl")
rows = sorted(problems.values(), key=lambda r:(r["tier"], r["subject"], int(r["id"])))
with open(out_path, "w") as f:
    for r in rows:
        f.write(json.dumps(r)+"\n")

# ---- 6. Summaries ----
print("Wrote", out_path)
print("TOTAL problems:", len(problems))
print("\n== Category splits ==")
bycat=defaultdict(int)
for r in problems.values(): bycat[r["subject"]]+=1
for c in sorted(bycat, key=lambda c:-bycat[c]): print(f"  {bycat[c]:4d}  {c}")

print("\n== MATH level distribution ==")
bylvl=defaultdict(int)
for r in problems.values(): bylvl[r["level"]]+=1
for l in sorted(bylvl, key=lambda x:(x is None, x)): print(f"  Level {l}: {bylvl[l]}")

print("\n== Difficulty tier distribution (T1 easy .. T5 hardest) ==")
byt=defaultdict(int)
for r in problems.values(): byt[r["tier"]]+=1
for t in sorted(byt): print(f"  T{t}: {byt[t]}")

print("\n== gpt4 functional snapshots-solved distribution ==")
byg=defaultdict(int)
for r in problems.values(): byg[r["gpt4_fn"]]+=1
for g in sorted(byg): print(f"  gpt4 solved {g}/3 snapshots: {byg[g]}")

print("\n== Tier x subject (rows=tier) ==")
tc=defaultdict(lambda: defaultdict(int))
for r in problems.values(): tc[r["tier"]][r["subject"]]+=1
subs=sorted(bycat)
print("tier " + " ".join(f"{s[:6]:>6}" for s in subs))
for t in sorted(tc):
    print(f"T{t}   " + " ".join(f"{tc[t][s]:>6}" for s in subs))

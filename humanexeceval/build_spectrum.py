r"""HumanExecEval pipeline, step 2/2.

Reads `full_tasks.jsonl` (from analyze.py), applies the selection filters and the
difficulty-stratified sampler, and writes the three deliverables next to this script:
  - humanexeceval_tasks_174.jsonl   (full metadata per task)
  - humanexeceval_tasks_174.csv     (scannable columns)
  - humanexeceval_tasks_174_io.md   (every task's 3 instantiated inputs + exact outputs)

Selection funnel (each filter operates on the 3 observed seed answers, since the
input/prb/sol source functions are unseen):
  1. drop tasks with no resolvable source in the .filelist manifest
  2. drop float-producers        -> inexact sol(), ill-posed "predict exact output"
  3. drop magnitude blow-ups      -> any integer literal (int OR fraction num/denom OR
                                     matrix entry) >= THRESHOLD_BIG, treated uniformly
Symbolic irrationals (\pi, \sqrt) and exact rational fractions of any size are KEPT
(exactly traceable by a human). Then sample TARGET tasks stratified across the 5 empirical
tiers, spread over subjects + MATH levels; rank by MATH level (primary), tier (secondary).

Run:  python3 humanexeceval/analyze.py && python3 humanexeceval/build_spectrum.py
"""
import json, os, glob, re, csv
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # repo root (snapshots, MATH/, *.filelist)

# A task's 3 seed answers ARE sol()'s outputs for 3 seeds. A decimal point (incl. a
# leading/trailing dot) or scientific notation => the hidden function does inexact float
# arithmetic. Exact rationals (\frac{}{}), radicals (\sqrt), and \pi expressions are kept.
FLOAT_RE = re.compile(r'\.\d|\d\.|\d[eE][+-]?\d')   # decimals incl. leading/trailing dot, + sci notation
def produces_float(seed_answers):
    return any(FLOAT_RE.search(a) for a in seed_answers)

# Magnitude blow-up: integers and fraction components are treated UNIFORMLY. Any integer
# literal in the answer -- standalone, a fraction numerator/denominator, or a matrix entry --
# reaching THRESHOLD_BIG for at least one seed marks the task as a blow-up.
# Commas are tuple separators (e.g. (45,54)) so they are NOT stripped/merged.
THRESHOLD_BIG = 10**6   # any integer literal >= 1e6 (7+ digits) counts as a blow-up
def _clean_for_scan(a):
    s = a.replace("$", "").replace(" ", "").replace("\\!", "").replace("\\,", "").replace("{,}", "")
    return re.sub(r"\^\\?\{?\\circ\}?", "", s).replace("\\%", "").replace("%", "")
def big_magnitude(seed_answers):
    for a in seed_answers:
        nums = [int(x) for x in re.findall(r"\d+", _clean_for_scan(a))]
        if nums and max(nums) >= THRESHOLD_BIG:
            return True
    return False

# NOTE: symbolic irrationals (\pi, \sqrt) are KEPT on purpose -- they are exact operations,
# fully traceable by a human. Written-out decimal expansions (e.g. 3.14159) are already
# removed by produces_float(). So no separate irrational filter is applied.

SNAPS = ["Oct-2023", "Nov-2023", "Dec-2023"]

# ---- load full analyzed table ----
tasks = [json.loads(l) for l in open(os.path.join(HERE, "full_tasks.jsonl"))]

# ---- source-file mapping (exact / suffixed / missing) ----
src_ids = defaultdict(list)
for line in open(f"{ROOT}/Oct-2023.filelist"):
    p = line.split(" = ")[0].strip().split("/")
    if len(p) == 3 and p[2].startswith("m") and p[2].endswith(".py"):
        name = p[2][1:-3]; base = name.split("_x")[0]
        src_ids[(p[1], base)].append(name)

def source_info(subj, pid):
    names = src_ids.get((subj, pid), [])
    if pid in names:
        return f"benchmarks/{subj}/m{pid}.py", "exact"
    elif names:
        return f"benchmarks/{subj}/m{sorted(names)[0]}.py", f"scaled-variant (candidates: {','.join('m'+n for n in sorted(names))})"
    else:
        return None, "no-source-in-filelist"

# attach source resolution to every task; drop unresolvable ones from the pool
for t in tasks:
    t["_src"], t["_src_note"] = source_info(t["subject"], t["id"])
n0 = len(tasks)
tasks = [t for t in tasks if t["_src_note"] != "no-source-in-filelist"]
# drop float-producing functions (inexact output => ill-posed "predict exact output" task)
n1 = len(tasks)
tasks = [t for t in tasks if not produces_float(t["seed_answers"])]
n2 = len(tasks)
# drop magnitude blow-ups (integers & fraction components treated uniformly; geometry/8's
# 12-digit numerator is caught here, so no manual exclusion is needed)
tasks = [t for t in tasks if not big_magnitude(t["seed_answers"])]
print(f"pool filter: {n0} -> {n1} (drop no-source) -> {n2} (drop float-producers: {n1-n2})"
      f" -> {len(tasks)} (drop magnitude blow-ups >= {THRESHOLD_BIG}: {n2-len(tasks)})")

# ---- load static MATH problem text ----
def static_problem(subj, pid):
    fp = f"{ROOT}/MATH/test/{subj}/{pid}.json"
    if os.path.exists(fp):
        return json.load(open(fp)).get("problem", "")
    return ""

TIER_LABEL = {
    1: "easy — gpt-4 solves all 3 functional snapshots (broad model agreement)",
    2: "moderate — gpt-4 solves >=2/3 functional snapshots",
    3: "hard — gpt-4 flaky (1/3) or only weaker models solve",
    4: "very hard — solved by exactly one model, <3 snapshots",
    5: "frontier-hard — no evaluated model solves it functionally",
}

# ---- deterministic diversity-aware stratified sampler: TARGET across 5 tiers ----
TARGET = 174   # ~10% of 1745 published tasks; sits beside HumanEval (164) -> "HumanExecEval"
base = TARGET // 5; rem = TARGET % 5
# uniform across the 5 empirical tiers; distribute the remainder to the harder tiers
TIER_QUOTA = {t: base for t in range(1, 6)}
for t in list(range(5, 0, -1))[:rem]:
    TIER_QUOTA[t] += 1

selected = []
for tier in range(1, 6):
    pool = [t for t in tasks if t["tier"] == tier]
    by_subj = defaultdict(list)
    for t in pool:
        by_subj[t["subject"]].append(t)
    # sort each subject's list to spread MATH levels: interleave levels present
    for s in by_subj:
        by_subj[s].sort(key=lambda t: (t.get("level") or 0, int(t["id"])))
    picked, level_count = [], defaultdict(int)
    # round-robin over subjects (ordered by inventory desc for fairness); each pick
    # chooses the candidate improving MATH-level balance the most.
    subj_cycle = sorted(by_subj.keys(), key=lambda s: -len(by_subj[s]))
    cursor = {s: 0 for s in by_subj}
    quota = TIER_QUOTA[tier]
    while len(picked) < min(quota, len(pool)):
        progressed = False
        for s in subj_cycle:
            if len(picked) >= min(quota, len(pool)):
                break
            lst = by_subj[s]
            if cursor[s] >= len(lst):
                continue
            remaining = lst[cursor[s]:]
            best = min(remaining, key=lambda t: (level_count[t.get("level")], int(t["id"])))
            lst.remove(best)
            picked.append(best); level_count[best.get("level")] += 1
            progressed = True
        if not progressed:
            break
    selected.extend(picked)

# ---- order final list by MATH level (primary) then empirical tier (secondary) ----
selected.sort(key=lambda t: ((t.get("level") or 0), t["tier"], t["subject"], int(t["id"])))

# ---- build records ----
records = []
for i, t in enumerate(selected, 1):
    src, note = t["_src"], t["_src_note"]
    records.append({
        "rank": i,
        "id": t["id"],
        "task_key": f"{t['subject']}/{t['id']}",
        "subject": t["subject"],
        "math_level": t["level"],                      # PRIMARY difficulty axis
        "empirical_tier": t["tier"],                   # SECONDARY (1 easy .. 5 hardest)
        "empirical_tier_label": TIER_LABEL[t["tier"]],
        "difficulty_label": f"L{t['level']} · empirical-T{t['tier']}",
        "source_file": src,
        "source_file_note": note,
        "empirical": {
            "gpt4_functional_snapshots_solved": t["gpt4_fn"],   # 0..3
            "gpt4_static_solved": bool(t["gpt4_static"]),
            "n_models_functional_any": t["n_models_fn_any"],    # 0..12
            "n_models_functional_robust_all3": t["n_models_fn_robust"],
            "n_models_static": t["n_models_static"],
            "ensemble_functional_snapshots": t["ens_fn_snapshots"],  # 0..36
            "solvability_0to1": t["solvability"],
            "reasoning_gap_flag": t["reasoning_gap_flag"],
        },
        "answer_complexity": {
            "answer_type": t["answer_type"],
            "exact_output": True,   # float-producers & magnitude blow-ups excluded at selection time
            "max_answer_len_across_seeds": t["max_answer_len"],
            "n_distinct_answers_across_3_seeds": t["n_distinct_answers"],
            "avg_problem_char_len": t["avg_problem_len"],
        },
        "seed_answers": t["seed_answers"],
        "static_math_problem": static_problem(t["subject"], t["id"]),
    })

# ---- emit deliverable 1: jsonl ----
jsonl_path = os.path.join(HERE, "humanexeceval_tasks_174.jsonl")
with open(jsonl_path, "w") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- emit deliverable 2: companion csv ----
csv_path = os.path.join(HERE, "humanexeceval_tasks_174.csv")
CSV_COLS = ["rank","task_key","subject","math_level","empirical_tier","empirical_tier_short",
            "source_file","source_file_note","gpt4_fn_snaps_solved","n_models_functional_any",
            "solvability_0to1","answer_type","n_distinct_answers_3seeds"]
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(CSV_COLS)
    for r in records:
        e = r["empirical"]; a = r["answer_complexity"]
        w.writerow([r["rank"], r["task_key"], r["subject"], r["math_level"], r["empirical_tier"],
                    r["empirical_tier_label"].split(" — ")[0], r["source_file"] or "",
                    r["source_file_note"].split(" ")[0], e["gpt4_functional_snapshots_solved"],
                    e["n_models_functional_any"], e["solvability_0to1"], a["answer_type"],
                    a["n_distinct_answers_across_3_seeds"]])

# ---- emit deliverable 3: full inputs/outputs markdown (all 3 seed instances per task) ----
md_path = os.path.join(HERE, "humanexeceval_tasks_174_io.md")
with open(md_path, "w") as f:
    f.write("# HumanExecEval — 174 tasks: full inputs & outputs\n\n")
    for r in records:
        subj, pid = r["task_key"].split("/")
        f.write(f"## {r['rank']}. `{r['task_key']}` — MATH Level {r['math_level']}, "
                f"empirical-T{r['empirical_tier']} ({r['empirical_tier_label'].split(' — ')[0]})\n")
        f.write(f"- source: `{r['source_file']}` | gpt-4 fn {r['empirical']['gpt4_functional_snapshots_solved']}/3 "
                f"| answer_type: {r['answer_complexity']['answer_type']}\n\n")
        for snap in SNAPS:
            d = json.load(open(f"{ROOT}/{snap}/test/{subj}/{pid}.json"))
            m = re.search(r"\\boxed\{(.*)\}", d["solution"], re.DOTALL)
            f.write(f"- **INPUT ({snap})**: {d['problem'].strip()}\n")
            f.write(f"  - **OUTPUT**: `{(m.group(1) if m else d['solution']).strip()}`\n")
        f.write("\n")

# ---- QA summary ----
print("TIER_QUOTA:", TIER_QUOTA, "sum:", sum(TIER_QUOTA.values()))
print(f"Wrote {len(records)} tasks ->\n  {jsonl_path}\n  {csv_path}\n  {md_path}")
def dist(key):
    d = defaultdict(int)
    for r in records: d[key(r)] += 1
    return dict(sorted(d.items()))
print("\nBy empirical tier:", dist(lambda r: f"T{r['empirical_tier']}"))
print("By MATH level:    ", dist(lambda r: f"L{r['math_level']}"))
print("By subject:       ", dist(lambda r: r["subject"]))
print("\nlevel x tier grid (count):")
grid = defaultdict(int)
for r in records: grid[(r["math_level"], r["empirical_tier"])] += 1
print("       T1 T2 T3 T4 T5")
for lv in range(1,6):
    print(f"  L{lv}  " + " ".join(f"{grid[(lv,t)]:2d}" for t in range(1,6)))
print("\nsource_file_note breakdown:", dist(lambda r: r["source_file_note"].split(" ")[0]))

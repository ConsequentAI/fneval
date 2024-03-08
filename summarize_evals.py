import argparse
import inspect
import sys
import os
import json
import shutil
from persist import Persist
from typing import List, Dict, Any, Set, Tuple
from eval.measure_math import Eval, Metrics, SUBJECTS
from eval.runner import mk_dir_safe

from evaluate import DEFAULT_EVAL_PICKLE_FILE, ModelSpec

COVERAGE_PC_TOO_LOW_IF_BELOW = 70
PC = lambda num, dim: 100.0 * (num / float(dim) if dim else 0.0)


def write_lines(lines: List[Any], prefix, ext = "txt"):
    frm = inspect.stack()[1][3]
    outfile = f"{prefix}.{frm}.{ext}"
    with open(outfile, 'w') as f:
        f.write("\n".join(lines))
    print(f'Written to {outfile}')


def solved_by_models_across_snapshots(frac_models, es: Dict[str, Eval], prefix):
    model_solved: Dict[str, List[str]] = { model: e.get_solved(only_static = False) for model, e in es.items() }
    solved: List[List[str]] = [s for _, s in model_solved.items()]

    counts: Dict[str, int] = {}
    def incr(s, d):
        if s not in d:
            d[s] = 0
        d[s] += 1
    for solns in solved:
        for soln in solns:
            incr(soln, counts)

    threshold = len(es) * frac_models
    above_threshold = [s for s, times in counts.items() if times >= threshold]

    solved_solns = sorted(above_threshold)
    write_lines(solved_solns, prefix + f'_{frac_models}')


def fraction_functionally_tested(static, func) -> Tuple[int, int]:
    static_correct = [tm.fname for tm in static.individuals if tm.correct]
    func_tested = [tm.fname for tm in func.individuals if tm.fname in static_correct]
    num, dim = len(func_tested), len(static_correct)
    return num, dim


def reasoning_gap(static, combined) -> Tuple[int, int]:
    st = PC(*static.accuracy())
    cb = PC(*combined.accuracy())
    return st-cb, st


def stat_solved_by_majority_models_across_snapshots(es: Dict[str, Eval], prefix, extra):
    solved_by_models_across_snapshots(0.5, es, prefix)


def stat_solved_by_all_models_across_snapshots(es: Dict[str, Eval], prefix, extra):
    solved_by_models_across_snapshots(1.0, es, prefix)


def stat_solved_statics(es: Dict[str, Eval], prefix, extra):
    solved: Dict[str, List[str]] = { model: e.get_solved(only_static = True) for model, e in es.items() }
    solved_solns = sorted(list(set(s for _, solns in solved.items() for s in solns)))
    write_lines(solved_solns, prefix)


def static_func_combined(model, evals) -> Tuple[Any, Any, Any, Any, str, str, bool]:
    static = evals.static_metrics
    func = evals.func_metrics
    verbose = False
    model_name = mk_dir_safe(model)
    combined, func_of_static = static.with_functional(func, model_name, verbose)
    gap = PC(*reasoning_gap(static, combined))
    gap_tag = f'{gap:.2f}%'
    pc_fn = PC(*fraction_functionally_tested(static, func))
    pc_fn_tag = f"{pc_fn:.2f}%"
    cover_too_low = pc_fn <= COVERAGE_PC_TOO_LOW_IF_BELOW
    warn_low_cover = pc_fn_tag + (" (too low)" if cover_too_low else "")
    return static, func, func_of_static, combined, gap_tag, warn_low_cover, cover_too_low


def stat_accuracy(es: Dict[str, Eval], prefix, extra):
    hr = '-' * 100
    accuracies: List[str] = []
    for model, evals in es.items():
        if not accuracies:
            legend = evals.static_metrics.stats_legend()
            accuracies.append(legend)
        static, func, func_sub_static, combined, gap, warn_low_cover, cover_too_low = static_func_combined(model, evals)
        if cover_too_low:
            print(f'Coverage too low ({warn_low_cover}) for {model}. Still writing raw data to file.')
        gap_warn = ' (please dont cite; coverage too low)' if cover_too_low else ''
        accuracies += [
                hr, model, hr,
                'static:', static.stats(),
                'functional:', func.stats(),
                'functional correct amongst static correct:', func_sub_static.stats(),
                'combined:', combined.stats(),
                'reasoning gap:', gap + gap_warn,
                'pc functionally tested:', warn_low_cover,
                'coverage low:', str(cover_too_low),
                hr]
    write_lines(accuracies, prefix)


def stat_csv_static_func(es: Dict[str, Eval], prefix, extra):
    force_write_low_covers = 'FORCE_WRITE_LOW_COVERS' in extra
    accuracies: List[str] = ["Model,Static,Func,Frac Func Tested,Static Hall,Func Hall,Gap,Fn Coverage,Correct Amongst Fn Tested"]
    for model, evals in es.items():
        static, func, func_sub_static, combined, gap, warn_low, cover_too_low = static_func_combined(model, evals)
        if cover_too_low:
            print(f'Coverage too low ({warn_low}) for {model}.')
            if not force_write_low_covers:
                continue
        st = PC(*static.accuracy())
        fn = PC(*combined.accuracy())
        st_h = PC(*static.hallucinations())
        fn_h = PC(*combined.hallucinations())
        pc_fn = PC(*fraction_functionally_tested(static, func))
        fn_correct = PC(*func_sub_static.accuracy())
        accuracies.append(f'{model},{st:.2f}%,{fn:.2f}%,{pc_fn:.2f}%,{st_h:.2f}%,{fn_h:.2f}%,{gap},{warn_low},{fn_correct:.2f}%')
    write_lines(accuracies, prefix, ext = "csv")


def stat_csv_subject_level(es: Dict[str, Eval], prefix, extra):
    # model -> typ (subject or level) -> (static, func)
    ModelTypStatFunc = Dict[str, Dict[Any, Tuple[float, float]]]

    def acc_fn(outcomes: List[bool]):
        return PC(outcomes.count(True), len(outcomes))

    def acc_s(m: Metrics, subject: str):
        return acc_fn(m.subject_cors[subject])

    def acc_l(m: Metrics, lvl: int):
        return acc_fn(m.level_cors[lvl])

    def collapse_models(metrics: ModelTypStatFunc, fn) -> ModelTypStatFunc:
        def collapse_tuple(l: List[Tuple[float, float]]) -> Tuple[float, float]:
            statics, funcs = list(zip(*l))
            return fn(statics), fn(funcs)
        models = list(m for m in metrics)
        types = list(t for t in metrics[models[0]])
        return { "all": { t: collapse_tuple(list(metrics[model][t] for model in models)) for t in types } }


    sub_accs: ModelTypStatFunc = {}
    lvl_accs: ModelTypStatFunc = {}
    for model, evals in es.items():
        static, _, _, comb, _, _, _ = static_func_combined(model, evals)
        s_accs = { sub: (acc_s(static, sub), acc_s(comb, sub)) for sub in SUBJECTS }
        l_accs = { lvl: (acc_l(static, lvl), acc_l(comb, lvl)) for lvl in range(1,6) }
        sub_accs[model] = s_accs
        lvl_accs[model] = l_accs

    def write(tag: str, accs: ModelTypStatFunc):
        lines: List[str] = [f"Model,{tag},Static,W Func,Delta"]
        for model, accuracies in accs.items():
            for typ, (stat, func) in accuracies.items():
                delta = PC(stat - func, stat)
                lines.append(f'{model},{typ},{stat:.2f}%,{func:.2f}%,{delta:.2f}%')
        write_lines(lines, prefix + f".{tag}", ext = "csv")

    write("Subject", sub_accs)
    write("Level", lvl_accs)

    avg = lambda xs: sum(xs)/len(xs)
    write("AllModels.Subject", collapse_models(sub_accs, avg))
    write("AllModels.Level", collapse_models(lvl_accs, avg))


def stat_dropoff(es: Dict[str, Eval], prefix, extra):
    tab = '\t'
    def summary_line(s, f):
        jsn = f'functional-math/prbs/{s.fname}'
        subject, base = s.fname.split('/')
        pyid = base[:-len('.json')]
        fnpy = f'functional-math/functional/benchmarks/{subject}/m{pyid}.py'
        lines = [
                s.answer,
                f.answer,
                f.output,
                f'{f.fraction_correct:.2f}',
                f'{s.level}',
                jsn,
                fnpy
        ]
        return tab.join(lines)
    hdr = tab.join([
        'static got (= correct)',
        'func correct ref',
        'func got (incorrect)',
        'fraction correct',
        'level',
        'json',
        'functional'
    ])

    def get_static_for(f, static):
        return next(tm for tm in static.individuals if tm.fname == f.fname)

    for model, evals in es.items():
        static, func, _, _, _, _, _ = static_func_combined(model, evals)
        dropped = [hdr]
        for fn in func.individuals:
            if not fn.correct:
                assert fn.output != fn.answer, 'aggregation failure? some functional incorrect, but diff outs not captured?'
                st = get_static_for(fn, static)
                if st.correct:
                    dropped.append(summary_line(st, fn))
        fname = f'dropped.{mk_dir_safe(model)}'
        write_lines(dropped, fname, ext = 'tsv')


def stat_single_model_contribution(es: Dict[str, Eval], prefix, extra):
    corrects = {}
    for model, evals in es.items():
        static, func, _, _, _, _, _ = static_func_combined(model, evals)
        corrects[model] = [s.fname for s in static.individuals if s.correct]
    cummulative: Dict[str, List[str]] = {}
    for model in es:
        for c in corrects[model]:
            if c not in cummulative:
                cummulative[c] = []
            cummulative[c].append(model)

    exclusives = [f'Across all models: {len(cummulative)}']
    only_in_model = {}
    for model in es:
        only_here = [c for c in cummulative if cummulative[c] == [model]]
        only_in_model[model] = only_here
        exclusives += [f'Exclusive to {model}: {len(only_here)}']
    write_lines(exclusives, prefix + '.exclusives')

    overlaps = [f'{c},{len(cummulative[c])}' for c in cummulative]
    write_lines(overlaps, prefix + '.count_models', ext='csv')


def stat_solved_statics_not_functionalized(es: Dict[str, Eval], prefix, extra):
    # root of already functionalized problems
    prb_root = extra['ROOT_FUNCMATH_PRBS']

    def read_functional_math_prbs() -> List[str]:
        files = []
        subjects = os.listdir(prb_root)
        for s in subjects:
            d = os.path.join(prb_root, s)
            for f in os.listdir(d):
                files.append(os.path.join(s, f))
        return files

    def copy_from_MATH(ufn: str, root: str):
        subj, ident = ufn.split('/')
        src = os.path.join('MATH', 'test', subj, ident)
        dst_dir = os.path.join(root, subj)
        dst = os.path.join(dst_dir, ident)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copyfile(src, dst)

    solved: Dict[str, List[str]] = { model: e.get_solved(only_static = True) for model, e in es.items() }
    solved_solns: List[str] = sorted(list(set(s for _, solns in solved.items() for s in solns)))
    prbs: List[str] = read_functional_math_prbs()
    unfunctionalized = [s for s in solved_solns if not s in prbs]

    # write output /prbs data to this directory
    outdir = prefix + '.not_yet_func'
    # read jsons and copy to directory structure with root of fn name
    ufn_root = os.path.join(outdir, 'prbs')
    if not os.path.exists(ufn_root):
        os.makedirs(ufn_root)
    for ufn in unfunctionalized:
        copy_from_MATH(ufn, ufn_root)
    print(f'Unfunctionalized MATH test jsons written to: {ufn_root}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file",
            help = f"Read pickle file of [ model_name -> eval ]; default = {DEFAULT_EVAL_PICKLE_FILE}")
    parser.add_argument("--stat_fn",
            help = f"Name of stat fn to run")
    parser.add_argument("--extra",
            help = f"Opt dict of other parameters")
    args = parser.parse_args()

    if not args.stat_fn:
        stat_names = [s for s in dir(sys.modules[__name__]) if s.startswith('stat')]
        print(f'--stat_fn can take: {stat_names}')
        exit(1)

    infile = args.in_file if args.in_file else DEFAULT_EVAL_PICKLE_FILE

    evals_spec: Dict[ModelSpec, Eval] = Persist.load(infile)
    evals: Dict[str, Eval] = { ms.ident(): e for ms, e in evals_spec.items() }
    print(f'Processing eval data from: {list(evals.keys())}')
    stat_fn = getattr(sys.modules[__name__], args.stat_fn)
    prefix = infile
    extra = json.loads(args.extra) if args.extra else {}

    # run the requested stat fn
    stat_fn(evals, prefix, extra)

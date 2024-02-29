from __future__ import annotations
import os
import csv
import json
from math_utils.math_helpers import clean_numbers, last_boxed_only # these are unused; why?
from math_utils.math_helpers import last_boxed_only_string, remove_boxed
from math_utils.math_equivalence import is_equiv
from helper_utils import NO_SOLUTION_PREFIX
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from fn_snapshot import FnSnapshot, root_test
from tqdm import tqdm

INTERMEDIATES_DIR = "intermediate"
PREFIX_FN_SOLVED = os.path.join(INTERMEDIATES_DIR, "functionals_solved_")
PREFIX_ANSWERS = os.path.join(INTERMEDIATES_DIR, "answers_")
PREFIX_CORRECT = os.path.join(INTERMEDIATES_DIR, "correct_")

if not os.path.exists(INTERMEDIATES_DIR):
    os.makedirs(INTERMEDIATES_DIR)

# these should match functional-math/functional/regression/instantiate.py
FN_SOLVED_STATIC_UNSOLVED = 'FN_SOLVED_STATIC_UNSOLVED'
FN_SOLVED_STATIC_SOLVED = 'FN_SOLVED_STATIC_SOLVED'

DELIM = "#" * 80 + "\n"
SUBJECTS = [
        'Prealgebra',
        'Algebra',
        'Number Theory',
        'Counting & Probability',
        'Geometry',
        'Intermediate Algebra',
        'Precalculus'
]


class TestMetric:
    def __init__(self, correct, frac_correct, output, completion, answer, typ, level, is_static, fname):
        # evaluation output and result
        self.correct = correct
        self.fraction_correct = frac_correct
        self.output = output
        self.answer = answer
        self.raw_completion = completion

        # identification data
        self.type = typ
        self.level = level
        self.is_static = is_static
        self.fname = os.path.join(*Path(fname).parts[-2:]) # `subject/id.json`

    def __repr__(self):
        s = f"CORRECT: {self.correct} | TYPE: {self.type} | LEVEL: {self.level} | OUTPUT: {self.output} | ANSWER: {self.answer} | STATIC: {self.is_static} | FNAME: {self.fname}"
        return s

    def __repr_concise__(self):
        s = f'{self.correct} | {self.output} | GT: {self.answer} | STATIC: {self.is_static} | {self.fname}'
        return s

    @classmethod
    def consensus(cls, many: List[TestMetric]) -> TestMetric:
        first = many[0]
        agree = lambda field: all(field(first) == field(m) for m in many)
        assert agree(lambda x: x.type), f'Differs on `type`: {many}'
        assert agree(lambda x: x.level), f'Differs on `level`: {many}'
        assert agree(lambda x: x.is_static), f'Differs on `is_static`: {many}'
        assert agree(lambda x: x.fname), f'Differs on `fname`: {many}'

        t, l, s, f = first.type, first.level, first.is_static, first.fname

        # if all correct then doesn't matter, we'll default to the first of the many
        assign = lambda x: (x.output, x.raw_completion, x.answer)
        valid, o, c, a = True, *assign(first)
        for m in many:
            if not m.correct:
                valid, o, c, a = False, *assign(m)
                break
        frac_valid = [m.correct for m in many].count(True) / float(len(many))
        return TestMetric(valid, frac_valid, o, c, a, t, l, s, f)


class Metrics:
    def __init__(self):
        self.individuals: List[TestMetric] = []
        self.name = None

        self.cors: Dict[Any, Any] = {}
        self.subject_cors: Dict[Any, Any] = {}
        self.level_cors: Dict[Any, Any] = {}
        self.correct = 0
        self.total = 0
        self.total_incorrect = 0
        self.hallucinated = 0

    def update_with(self, rslt: TestMetric):
        self.individuals.append(rslt)
        equiv = rslt.correct
        prob_level = rslt.level
        prob_type = rslt.type

        # update subject and level corrects
        if (prob_level, prob_type) in self.cors:
            self.cors[(prob_level, prob_type)].append(equiv)
        else:
            self.cors[(prob_level, prob_type)] = [equiv]
        if prob_level in self.level_cors:
            self.level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                self.level_cors[prob_level] = [equiv]
        if prob_type in self.subject_cors:
            self.subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                self.subject_cors[prob_type] = [equiv]

        # update correct/total counts
        if equiv:
            self.correct += 1
        self.total += 1

        # update hallucinations count
        if not equiv:
            lowered_answer = rslt.output.lower().strip()
            no_soln_prefix = NO_SOLUTION_PREFIX.lower()
            says_dont_know = lowered_answer.startswith(no_soln_prefix)
            if not says_dont_know:
                self.hallucinated += 1
            self.total_incorrect += 1

    def accuracy(self):
        return self.correct, self.total

    def hallucinations(self):
        return self.hallucinated, self.total_incorrect

    def stats_legend(self):
        return "[A]ccuracy, [H]allucination"

    def stats(self):
        def stat_str(num: int, dim: int) -> str:
            pc = 100.0 * (num / float(dim) if dim else 0.0)
            return f'{num:04d}/{dim:04d} = {pc:5.2f}%'

        return     f"A: {stat_str(*self.accuracy())}"\
                f" | H: {stat_str(*self.hallucinations())}"

    def save_and_log(self, tested_functionally: Dict[TestMetric, TestMetric], model_name: str, verbose):
        console = []
        filelog = []

        for s, f in tested_functionally.items():
            # default - functional unsolved; static solved; or both unsolved
            log: Optional[str] = None
            row: Optional[Any] = None
            if f.correct:
                if s.correct:
                    log = f'[INFO] TOO SIMPLE? - functional solved: {s.fname} at Level {s.level}'
                    row = [s.fname, FN_SOLVED_STATIC_SOLVED, s.level]
                else:
                    log = f'[WARN] UNEXPECTED - functional solved; static unsolved: {s.fname}'
                    row = [s.fname, FN_SOLVED_STATIC_UNSOLVED, s.level]
            console.append(log)
            filelog.append(row)

        solved_fns_file = f'{PREFIX_FN_SOLVED}{model_name}.csv'
        with open(solved_fns_file, 'w') as cf:
            csvwriter = csv.writer(cf)
            for l in filelog:
                if l is not None:
                    csvwriter.writerow(l)

        if verbose:
            console_lines = "\n".join(c for c in console if c is not None)
            print(f'{console_lines}')
            print(f'written solved functionals to {solved_fns_file}')

            # print stats
            cases = [case for _, case, _  in [row for row in filelog if row]]
            s_f_correct = cases.count(FN_SOLVED_STATIC_SOLVED)
            s_incorrect_f_correct = cases.count(FN_SOLVED_STATIC_UNSOLVED)
            print(f'[STATS] tested functionally: {len(tested_functionally)}, correct f+s: {s_f_correct} (counts towards accuracy), correct f+!s: {s_incorrect_f_correct} (does not count)')

    def get_overriden(self, func: Metrics, verbose: bool) -> Dict[TestMetric, Optional[TestMetric]]:
        if verbose:
            statics = set(f.fname for f in func.individuals if f.is_static)
            print(f'[STATS] statics in the functional set: {len(statics)} (if 0 then snapshot might have been taken without include_static = True)')

        matching = lambda ident: [f for f in func.individuals if f.fname == ident and not f.is_static]
        overriden = {}
        for s in self.individuals:
            matches = matching(s.fname)
            if len(matches) > 1:
                print(f'More than one functional for {s.fname}. Picking first')
            overriden[s] = matches[0] if matches else None
        return overriden

    def with_functional(self, func: Metrics, model_name: str, verbose: bool) -> Metrics:
        overriden: Dict[TestMetric, Optional[TestMetric]] = self.get_overriden(func, verbose)
        with_fn = Metrics()
        tested_functionally: Dict[TestMetric, TestMetric] = {}
        total_correct, functionalized_correct = 0, 0
        f_none_count_s_correct = 0
        for s, f in overriden.items():

            if f is None:
                # update accumulator: only static available, so keep that
                with_fn.update_with(s)
                if s.correct:
                    f_none_count_s_correct += 1

            else:
                # update accumulator: both s & f available
                #   - case: both correct or both false, storing s/f is identical
                #   - case: one of them is false, keep the false one; since we want to be strict
                # summary: if s is incorrect store that, otherwise store whatever f's outcome was
                if not s.correct:
                    with_fn.update_with(s)
                else:
                    with_fn.update_with(f)

                # for ones overriden, put them in tested_functionally, so we can log to console and file
                tested_functionally[s] = f

        if verbose:
            print(f'[STATS] statics correct, for which no functional: {f_none_count_s_correct} (counts towards accuracy)')

        with_fn.name = f'{self.name}\\{func.name}'
        self.save_and_log(tested_functionally, model_name, verbose)

        return with_fn

    @classmethod
    def consensus(cls, many: List[Metrics]) -> Metrics:
        find = lambda fname, m: next(tm for tm in m.individuals if tm.fname == fname)
        idents = lambda i: set(tm.fname for tm in many[i].individuals)
        ref_idents = idents(0)
        assert all(ref_idents == idents(i) for i in range(1, len(many))), f'idents differ'
        by_name: Dict[str, List[TestMetric]] = { f: [find(f, m) for m in many] for f in ref_idents }
        cs: Dict[str, TestMetric] = { f: TestMetric.consensus(ltm) for f, ltm in by_name.items() }

        aggr = Metrics()
        aggr.name = "+".join([f'{m.name}' for m in many])
        for c in cs.values():
            aggr.update_with(c)
        return aggr

class Eval:
    def __init__(self, solve_fn, fn_snapshots, model_name, verbose = False):
        self.verbose = verbose
        self.static_metrics, self.func_metrics = self.get_metrics(solve_fn, fn_snapshots, model_name)

        # report combined metrics
        combined = self.static_metrics.with_functional(self.func_metrics, model_name, verbose)
        self.report_stats(combined)

        if not os.path.exists(INTERMEDIATES_DIR):
            os.makedirs(INTERMEDIATES_DIR)

    def get_metrics(self, solve_fn, fn_snapshots, model_name) -> Tuple[Metrics, Metrics]:
        static_metrics = None
        multiple_func_metrics = []
        for snapshot in fn_snapshots:
            # download and setup directories `ROOT/test/subject/id.json`
            root = snapshot.ensure_dataset()

            # location
            test_dir = root_test(root)
            tag = snapshot.date
            out_answers_to = f'{PREFIX_ANSWERS}{model_name}-{tag}'
            correct_dir = f'{PREFIX_CORRECT}{model_name}-{tag}'

            # eval over snapshot
            metrics = self.run_eval(tag, test_dir, solve_fn, out_answers_to)

            # log to data dirs
            self.write_correct_completions(metrics, correct_dir)

            # keep static aside, and put all functionals into list
            if snapshot.is_static():
                static_metrics = metrics
            else:
                multiple_func_metrics.append(metrics)

        func_metrics = Metrics.consensus(multiple_func_metrics)
        assert static_metrics, f'No static metrics!'
        return static_metrics, func_metrics

    def get_solved(self, only_static: bool = False) -> List[str]:
        names = lambda mm: set(tm.fname for tm in mm.individuals if tm.correct)
        s, f = names(self.static_metrics), names(self.func_metrics)
        return list(s if only_static else s.intersection(f))

    def report_stats(self, m: Metrics):
        print(DELIM)
        print(f'Legend: {m.stats_legend()}')
        print(f'{m.name}: {m.stats()}')
        print(DELIM)

    def run_eval(self, name, root, solve_fn, out_answers_file):
        metrics = self.eval_test_files(root, solve_fn)
        metrics.name = name
        self.write_results_file(metrics, out_answers_file)
        return metrics

    def eval_test_files(self, root, solve_fn) -> Metrics:
        m = Metrics()

        for subdir, dirs, files in os.walk(root):
            enum_files = files if self.verbose else tqdm(files, desc = subdir)
            for file in enum_files: # type: ignore

                fname = os.path.join(subdir, file)
                with open(fname, 'r') as fp:
                    self.eval_test_data(fname, fp, m, solve_fn)
        return m

    def compare_answer(self, ref, solns: List[Tuple[str, str]]) -> Tuple[bool, str, str]:
        def eq(t):
            try:
                return is_equiv(t, ref)
            except:
                return False

        def pick_best(many_solns: List[Tuple[bool, str, str]]) -> Tuple[bool, str, str]:
            return many_solns[0]

        equivs = [(eq(t), t, c) for t, c in solns]
        return pick_best(equivs)

    def eval_test_data(self, fname, fp, m, solve_fn) -> None:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {fname}", e)
            raise e
        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None
        prb = problem_data["problem"]
        soln = problem_data["solution"]
        ref_answer = remove_boxed(last_boxed_only_string(soln))
        is_static = problem_data["is_static"] if "is_static" in problem_data else True

        # ask model to solve problem
        solns: List[Tuple[str, str]] = solve_fn(prb)
        equiv, text_answer, completion = self.compare_answer(ref_answer, solns)
        frac_correct = 1.0 if equiv else 0.0

        rslt = TestMetric(
                correct = equiv,
                frac_correct = frac_correct,
                output = text_answer,
                completion = completion,
                answer = ref_answer,
                typ = prob_type,
                level = prob_level,
                is_static = is_static,
                fname = fname,
        )
        m.update_with(rslt)

        running = m.stats() # percentages accuracy/hallucinations
        test_summary = rslt.__repr_concise__()
        if self.verbose:
            print(f'{running} | {test_summary}')

    def out(self, f, line):
        f.write(line + "\n")
        if self.verbose:
            print(line)

    def write_each_test_results(self, m, f):
        for k, t in enumerate(m.individuals):
            line = f"{k} {t}"
            self.out(f, line)

    def pc(self, cors_list):
        return np.sum(cors_list), len(cors_list), 100.0 * np.mean(cors_list)

    def write_level_accuracies(self, m, f):
        self.out(f, DELIM)
        for level in sorted(m.level_cors):
            if level not in m.level_cors.keys():
                continue
            cors_list = m.level_cors[level]
            line = "Level {} Accuracy = {}/{} = {:5.2f}%".format(level, *self.pc(cors_list))
            self.out(f, line)
        self.out(f, DELIM)

    def write_subject_accuracies(self, m, f):
        self.out(f, DELIM)
        for subject in SUBJECTS:
            if subject not in m.subject_cors.keys():
                continue
            cors_list = m.subject_cors[subject]
            line = "{} Accuracy = {}/{} = {:5.2f}%".format(subject, *self.pc(cors_list))
            self.out(f, line)
        self.out(f, DELIM)

    def write_subject_level_accuracies(self, m, f):
        self.out(f, DELIM)
        for subject in SUBJECTS:
            for level in range(1, 6):
                key = (level, subject)
                if key not in m.cors.keys():
                    continue
                cors_list = m.cors[key]
                line = "{} Level {} Accuracy = {}/{} = {:5.2f}%".format(subject, level, *self.pc(cors_list))
                self.out(f, line)
        self.out(f, DELIM)

    def write_correct_completions(self, m, dirpath: str):
        print(f'Writing {m.name} raw+correct completions to {dirpath}')
        # make directory if it doesn't exist
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # for each test outcome, if not correct continue,
        # if correct then write raw to file in dump directory
        for t in m.individuals:
            if not t.correct:
                continue

            ident, json = os.path.splitext(os.path.basename(t.fname))
            subject = os.path.split(t.fname)[0]
            loc = os.path.join(dirpath, subject)
            if not os.path.exists(loc):
                os.makedirs(loc)
            fname = os.path.join(loc, f'{ident}.txt')
            raw_completion = t.raw_completion
            with open(fname, 'w') as f:
                f.write(raw_completion)

    def write_results_file(self, m: Metrics, out_answers_file: str):
        with open(out_answers_file, "w+") as f:
            # write all the raw test results
            self.write_each_test_results(m, f)

            # write accuracies by subject/level
            self.write_subject_level_accuracies(m, f)
            self.write_level_accuracies(m, f)
            self.write_subject_accuracies(m, f)

            # write overall accuracy
            self.out(f, f'Overall {m.name}: {m.stats()}')


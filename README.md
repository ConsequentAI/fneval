
# Functional Benchmarks and Reasoning Gap

This repository accompanies the paper
[Functional Benchmarks for Robust Evaluation of Reasoning Performance,
and the Reasoning Gap](https://arxiv.org/abs/2402.19450)

Summary tweet threads:
[Without CoT](https://twitter.com/_saurabh/status/1763626711407816930),
and [with CoT](https://twitter.com/_saurabh/status/1769852207925805080).

---

**Note (Feb'24): This repo and associated paper will not be finalized until the
Q2'24 release. We are working to get 100% coverage over MATH and GSM8K. We are
releasing the first version (Q1'24) for early access to the community.**

# Overview
We propose a way to evaluate reasoning capabilities of models that could have
been trained on the entire (compressed on uncompressed) text of the internet,
and possibly retrieval augmented.

We rewrite each benchmark question, in parameterized functional form, such that
the reasoning involved in solving each instance is identical, but the question
and answer are textually distinct each time. We snapshot a new instance of the
entire benchmark each month. A model has to solve the last K instances
correctly to count as correctly reasoning through that question. We notate
these functionalized versions of the benchmarks with "()", e.g., "MATH()" for
the "MATH" benchmark.

For each model the evaluation script tabulates:
* Static accuracy: accuracy on the static benchmark (e.g., MATH).
* Functional accuracy: accuracy over the last K instances of the functional benchmark
  (e.g., MATH()) and the static benchmark.
* Reasoning Gap: Percent drop from static to functional.
* Hallucination [0,100]: Percent times incorrect solution output, instead
  of stating no solution.

# Run (with external API calls)
Export the following keys:
```
export OPENAI_ORG="org-..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-..."
export TOGETHER_API_KEY="..."
export MISTRAL_API_KEY="..."
export HUGGING_FACE_HUB_TOKEN="hf_..."
```

After exporting the API keys, run:
```
python3 -m evaluate

# compute accuracies for static/functional (or run with just --stat_fn to see all available stats)
python3 -m summarize_evals --stat_fn stat_accuracy
```

# Run (offline, with manual model completions)
If you have an internal model that cannot be called via API, then it might be best to dump out all prompts to JSON, and complete them offline and bring them back into the harness. There is a fake api called `model_api/offline.py` that lets you do that.

Create two model spec JSONs, one for writing prompts, and one for reading completions

Writing prompts `evaluated_models_write_qs.json` should look like:
```
[ { "name": "your-local-model", "script": "model_api/offline.py", "include": true, "CoT": false, "extra_params": "{\"write_qs\": true}" } ]
```
Reading completions `evaluated_models.json` should look like:
```
[ { "name": "your-local-model", "script": "model_api/offline.py", "include": true, "CoT": false } ]
```

```
# dump prompts
python -m evaluate --model_specs evaluated_models_write_qs.json

# run your local model and complete each of the prompts in the output JSON
# create a new JSON by adding the completions as a "completion" field in the prompts JSON

# run completions
python3 -m evaluate --model_specs evaluated_models.json # this step is not fully implemented in the code

# compute accuracies for static/functional (or run with just --stat_fn to see all available stats)
python3 -m summarize_evals --stat_fn stat_accuracy
```

# FAQs

1. Why do the accuracy numbers differ from the best reported for the models?
We have not optimized for getting the best reported accuracy for each model,
and we likely will not do that. The best reported might need undisclosed
prompting, or post-processing.
We conjecture that the overall conclusions about reasoning gap presence
will not change.

1. Open question: pass@k and maj@k increase topline performance, but do they
   increase or decrease the reasoning gap?
Work-in-progress. Requires 100% MATH coverage.

1. Why is the GPT4 number not 78.2% as reported in the [Let's why step by
   step](https://arxiv.org/abs/2305.20050) paper?
As mentioned in their repository section [MATH
Splits](https://github.com/openai/prm800k#math-splits) and in the paper, their
evaluation is non-standard: "In order to avoid the risk of over-fitting on the
7,500 MATH training problems, we expanded the training set to include 4,500
MATH test split problems. We therefore evaluate our models only on the
remaining 500 held-out problems. We selected these 500 test problems uniformly
at random, and we believe they are representative of the test set as a whole."
For the 500 representative problems they pick PRM, ORM, and Majority Voting get
78.2%, 72.4% and 69.6% when using best-of-1860 (Figure 3).

# Known Issues
* More liberal output math equivalence matching. E.g., `C(21,7)` instead of `116280`
  (Claude non-CoT); `9 choose 2 = 36` instead of `36` (Mixtral non-CoT).
* Functional instantiations can sometimes result in problems or answers that
  are too long. E.g., in one of the instantiations in the `Dec-2023` snapshot a
benchmark's output number has more than 40 digits.

# Citation

    @misc{srivastava2024functional,
          title={Functional Benchmarks for Robust Evaluation of Reasoning Performance, and the Reasoning Gap},
          author={Saurabh Srivastava and Annarose M B and Anto P V au2 and Shashank Menon and Ajay Sukumar and Adwaith Samod T and Alan Philipose and Stevin Prince and Sooraj Thomas},
          year={2024},
          eprint={2402.19450},
          archivePrefix={arXiv},
          primaryClass={cs.AI}
    }

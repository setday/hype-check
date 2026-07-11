# Hypecheck

Challenging Causal Foundation Models for Uplift 

## Description

This project examines the reliability of causal foundation models for uplift models. CausalPFN produces training-free estimates of the conditional average treatment effect (CATE), yet uplift modeling is a ranking problem, and accurate CATE estimation does not guarantee that individuals are correctly ordered by their incremental effect. We assess whether an untuned CausalPFN matches tuned learners on the Qini metric, characterize the regimes in which this advantage degrades, and introduce CausalPFN-Rank, a training-free adaptation that combines context selection with output re-ranking to recover the targeting performance lost in those regimes. Evaluation is carried out on real-world marketing randomized controlled trials.	We will use real marketing RCTs and semi-synthetic CATE benchmarks to evaluate whether an untuned CausalPFN matches tuned uplift learners on Qini, and map the regimes where this advantage fails. We will then develop our own model, levereging best of the studied approaches. 

## Scope

Uplift is a well-known ranking and policy decision task based on effect estimation (in comparison to the classic ML which just predicts single outcome). We investigate whether a ready-to-use causal foundation model (CausalPFN: [paper](https://arxiv.org/abs/2506.07918), [github](https://github.com/vdblm/CausalPFN/)) helps real targeting in three steps:

- Check: untuned causal foundation models vs strong tuned baselines on the ranking metrics;

- Break: map the regimes where foundation models advantages disappear;

- Fix: develop a CausalPFN-Rank, a training-free adaptation of the studied foundation models (smart context + output re-ranking).

> [!NOTE]
> `Why this is practically interesting?`
>
> 1. In business context uplift is judged by ranking (e.g., Qini, uplift@k, AUUC), but CausalPFN is tuned for effect error metrics -- we expect that the best estimator is not the best ranking model.
> 
> 1. Practical insights regarding when an untuned foundation model is enough vs the tuned one / specialized method / untuned + cheap fix.


## Main goal

We will use real marketing RCTs and semi-synthetic CATE benchmarks to evaluate whether an untuned CausalPFN matches tuned uplift learners on Qini, and map the regimes where this advantage fails. We will then develop our own model, levereging best of the studied approaches. 

## Plan

### Hypotheses

[~] H1  Untuned, CausalPFN matches tuned non-foundation learners on semi-synthetic accuracy (PEHE) but not on real targeting (Qini) — on strong-signal RCTs it trails tuned learners on Qini

[ ] H2  Ranking by accuracy correlates weakly with ranking by targeting value; the best estimator is often not the best targeter

[ ] H3  The advantage breaks in three regimes: data beyond the context window, a small control group, and low conversion

[ ] H4  Engineered context (treatment-balanced selection plus multi-context averaging) recovers much of the lost targeting value, with no training

[ ] H5  Re-ranking the outputs improves targeting even when accuracy is unchanged, narrowing the gap to specialist methods

### Sub goals

[~] G1: reproducible uplift results over the dataset pool; reproduce CausalPFN's Hillstrom example as a smoke test; — harness + S/T/X/DR + CausalPFN over all five datasets

[ ] G2: map the break-axes by controlled subsampling, with bootstrap CIs and the accuracy-vs-targeting correlation;

[ ] G3: build and ablate CausalPFN-Rank; report per- regime recovery and compute cost.

### Tasks

[~] W1: harness, smoke test, baselines, first untuned comparison; — `scripts/run_baselines.py`

[ ] W2: break-axis mapping; add Q-Learner and GP-CATE;

[ ] W3: implement and ablate CausalPFN-Rank;

[ ] W4: final runs with CIs and efficiency; write-up.

## Notes

> [!NOTE]
> `CausalPFN-Rank (frozen model, no training)`
>
> - Input: treatment-balanced context and averaging over several retrieved contexts, vs the model's own default retrieval;
>
> - Output: re-rank the scores, a calibrator trained for ordering, a cautious lower-bound score, and a ratio score for low conversion.

> [!NOTE]
> `Baselines, data, metrics`
>
> - Baselines: S/T/X/DR on boosting, Causal Forest, BCF, DragonNet/DESCN; Q-Learner and GP-CATE on their regimes;
>
> - Data*: [Criteo](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) / [X5](https://www.uplift-modeling.com/en/v0.4.1/api/datasets/fetch_x5.html) and [Hillstrom](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) (real RCT); [IHDP](https://github.com/AMLab-Amsterdam/CEVAE) / [ACIC](https://github.com/BiomedSciAI/causallib/tree/master/causallib/datasets/data/acic_challenge_2016) (semi-synthetic);
>
> - Metrics: Qini/AUUC and uplift@k with bootstrap CIs; PEHE where ground truth exists; efficiency reported separately.

## Running the experiments

```bash
export HYPECHECK_DATA_ROOT=/path/to/data_A_cleaned

python scripts/run_baselines.py --datasets all --limit 300000
python scripts/run_baselines.py --datasets all --with-causalpfn --eval-limit 6000 --outdir results_with_pfn
python scripts/smoke_test.py --dataset hillstrom --models s_learner,causalpfn
```

Results (`metrics.csv`, `metrics.md`, `qini_*.png`) are written to `--outdir` (default `results/`).
Precomputed W1 results (`results/`, `results_with_pfn/`) are on [Google Drive](https://drive.google.com/drive/folders/1RJrAvz2cbam-1YAwcigFabHDDnAXwPFl?usp=sharing).

## Materials

https://arxiv.org/abs/2506.07918
https://arxiv.org/abs/2410.07021v1
https://arxiv.org/abs/2605.26288
https://arxiv.org/abs/2605.27473

# TP_Markov2026

Submission-ready Python project for the "Markov Chains for Text Generation" lab. This repository contains the full codebase, cached data files, generated outputs, and a report-style summary so the project can be reviewed directly from GitHub.

## 1. Project Objective

The goal of this project is to build and evaluate Markov-chain language models for text generation, with a focus on:

- data acquisition from public text sources
- text preprocessing into a restricted character vocabulary
- first-order character-level modeling
- higher-order character-level modeling
- model evaluation with log-likelihood and perplexity
- text generation with full sampling, top-k sampling, and greedy decoding
- optional bonus word-level modeling

The project is implemented as a clean Python package and runnable scripts. No notebook is required.

## 2. Repository Structure

```text
TP_Markov2026/
|- .gitignore
|- README.md
|- requirements.txt
|- main.py
|- run_experiments.py
|- report_notes.md
|- data/
|  |- raw/
|  `- processed/
|- outputs/
|  |- samples/
|  |- metrics/
|  `- figures/
`- src/
   |- __init__.py
   |- config.py
   |- utils.py
   |- part1_scraping.py
   |- part2_preprocessing.py
   |- part3_order1_model.py
   |- part4_scoring.py
   |- part5_generation.py
   |- part6_orderN_model.py
   `- part7_wordlevel.py
```

## 3. How To Run

### Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Run the full experiment

```powershell
python run_experiments.py
```

The command above runs the complete pipeline from the project root and saves metrics and generated samples into `outputs/`.

## 4. Data Sources

The default configuration uses Project Gutenberg URLs:

- train: `https://www.gutenberg.org/cache/epub/11/pg11.txt`
- test_same: `https://www.gutenberg.org/cache/epub/12/pg12.txt`
- test_diff: `https://www.gutenberg.org/cache/epub/1342/pg1342.txt`

These defaults are defined in `src/config.py`.

### Important note for this submitted run

During execution in the current environment, direct network access to Gutenberg was blocked. Because of that, the pipeline automatically used local fallback excerpts and cached them into:

- `data/raw/train.txt`
- `data/raw/test_same.txt`
- `data/raw/test_diff.txt`

This fallback behavior is intentional and implemented so the project remains runnable and reproducible even when external fetching fails.

## 5. Preprocessing

The restricted character vocabulary is:

```python
['^', '$', ' '] + list('abcdefghijklmnopqrstuvwxyz')
```

Vocabulary size:

```text
29
```

Preprocessing rules implemented in `src/part2_preprocessing.py`:

- convert text to lowercase
- keep only letters `a-z` and spaces
- remove all other characters
- collapse repeated whitespace into a single space
- strip leading and trailing spaces
- add `^` at the beginning and `$` at the end
- raise `ValueError` if no valid content remains

For higher-order models of order `n`, preprocessing uses repeated start markers:

- order 1: `^text$`
- order 3: `^^^text$`
- order 5: `^^^^^text$`

## 6. Implemented Models

### 6.1 First-Order Character Model

Implemented in `src/part3_order1_model.py`.

This model counts transitions:

```text
current_char -> next_char
```

Laplace smoothing is applied:

```text
P(y|x) = (count(x->y) + 1) / (sum_z count(x->z) + |V|)
```

The implementation ensures:

- every vocabulary symbol has a probability distribution
- each row sums to 1
- the model is verified before use

### 6.2 Higher-Order Character Model

Implemented in `src/part6_orderN_model.py`.

This model:

- supports arbitrary order `n`
- uses tuple histories as dictionary keys
- applies Laplace smoothing
- stores distributions for seen histories
- uses a safe fallback distribution for unseen histories

The submitted run compares orders:

- order 1
- order 3
- order 5

### 6.3 Bonus Word-Level Model

Implemented in `src/part7_wordlevel.py`.

This optional model:

- tokenizes lowercase words with regex
- keeps a frequent-word vocabulary
- replaces rare words with `<UNK>`
- adds `<START>` and `<END>`
- builds a Laplace-smoothed word bigram model

## 7. Evaluation Method

Implemented in:

- `src/part4_scoring.py` for order-1
- `src/part6_orderN_model.py` for higher-order models

### Log-likelihood

Evaluation uses the sum of log probabilities:

```text
log_likelihood = sum(log P(next | history))
```

### Perplexity

```text
perplexity = exp(-log_likelihood / n_transitions)
```

Safe fallback probabilities are used when a transition or history is missing unexpectedly.

## 8. Main Experimental Results

The following results come from the saved output files in `outputs/metrics/`.

### 8.1 First-order model

| Split | Transitions | Log-likelihood | Avg log-likelihood | Perplexity |
|---|---:|---:|---:|---:|
| Train | 515 | -1211.4800 | -2.3524 | 10.5106 |
| Same style | 310 | -794.0787 | -2.5615 | 12.9558 |
| Different style | 286 | -782.0635 | -2.7345 | 15.4019 |

Interpretation:

- the first-order model performs best on the training text
- perplexity increases on held-out text
- perplexity is higher on different-style text than on same-style text, which is consistent with expectations

### 8.2 Order comparison

| Order | Train perplexity | Same-style perplexity | Different-style perplexity | Seen states | Possible states | Sparsity |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10.5106 | 12.9558 | 15.4019 | 24 | 29 | 0.827586 |
| 3 | 12.3724 | 20.5901 | 24.6378 | 318 | 24389 | 0.013039 |
| 5 | 14.3158 | 26.4130 | 28.6581 | 447 | 20511149 | 0.000022 |

Interpretation:

- on this small submitted dataset, higher-order models became much sparser
- because of sparsity, order 3 and order 5 did not outperform order 1 on perplexity
- this is a useful result rather than a failure: it shows that increasing order alone is not enough without sufficient training data

## 9. Frequent Transitions In The Order-1 Model

Top 10 transitions observed in the training data:

| From | To | Count |
|---|---|---:|
| `e` | ` ` | 19 |
| `h` | `e` | 16 |
| `i` | `n` | 15 |
| ` ` | `t` | 13 |
| `d` | ` ` | 12 |
| `e` | `r` | 12 |
| `t` | `h` | 12 |
| ` ` | `o` | 11 |
| `r` | ` ` | 11 |
| ` ` | `a` | 9 |

These are plausible English character transitions such as `th`, `he`, and word-boundary transitions involving spaces.

## 10. Generation Results

Generation supports:

- full sampling
- top-k sampling
- greedy decoding

### Observations from the submitted run

- full sampling produced noisy but nontrivial character sequences
- greedy decoding collapsed into repetitive patterns such as `the the the ...`
- higher-order top-k generation often terminated too early on the small fallback dataset

These behaviors are expected:

- greedy decoding often degenerates into repetition
- very small training data plus restricted vocabulary can lead to short or empty samples
- higher-order models become sparse quickly, which harms generation quality

### Sample outputs

Order-1 full sampling example:

```text
picobkwathes otuphngrfanxg tkmtalyetjolhg ad asiersicxpjbm wzpgmumad auutrssad blhug zle otfd wzmzvepmlie hoxwoupvvaymifigd s
```

Order-1 greedy example:

```text
as the the the the the the the the the the the the the the the the the the the the ...
```

Word-level sample example:

```text
picking the be considering sleepy pleasure very hot
```

All generated samples are saved in:

- `outputs/samples/order1_samples.md`
- `outputs/samples/order3_topk_samples.md`
- `outputs/samples/order5_topk_samples.md`
- `outputs/samples/wordlevel_samples.txt`

## 11. Mapping Between Requirements And Files

| Requirement | Implemented in |
|---|---|
| Data acquisition | `src/part1_scraping.py` |
| Preprocessing | `src/part2_preprocessing.py` |
| Order-1 model | `src/part3_order1_model.py` |
| Scoring and perplexity | `src/part4_scoring.py` |
| Sampling and generation | `src/part5_generation.py` |
| Higher-order model | `src/part6_orderN_model.py` |
| Word-level bonus | `src/part7_wordlevel.py` |
| End-to-end pipeline | `main.py` |
| Run script | `run_experiments.py` |

## 12. Output Files For Grading

The professor can directly inspect the following files:

- `data/raw/` for cached input texts
- `data/processed/` for cleaned texts with markers
- `outputs/metrics/order1_metrics.json` for first-order evaluation
- `outputs/metrics/order_comparison.json` for order comparison
- `outputs/metrics/top_transitions_order1.json` for the top transitions
- `outputs/metrics/summary.json` for the combined experiment summary
- `outputs/metrics/report_summary.md` for a compact markdown summary
- `outputs/samples/` for generated text samples

## 13. Relation To Modern Language Models

This project is deliberately simple compared with large language models:

- Markov models use only a fixed short context
- LLMs can use much longer context and richer learned representations
- both can still be discussed using generation quality and perplexity

This makes Markov chains useful as an educational baseline for understanding sequential probability modeling.

## 14. Limitations

- the restricted vocabulary removes punctuation, digits, and capitalization
- the submitted run used fallback local excerpts because network access was blocked
- the training dataset is small, which hurts higher-order models
- higher-order generation is sparse and often stops early
- character-level outputs capture local patterns better than global coherence

## 15. Final Submission Checklist

- code is organized as runnable Python scripts
- `python run_experiments.py` works from the project root
- `data/` is included in the repository
- `outputs/` is included in the repository
- order-1 and higher-order models are implemented
- metrics and generated samples are saved for review

This repository is intended to be sufficient both as executable code and as a readable submission artifact for grading.

# Fabricating Certainty: Analysis of Epistemic Humility through Automated Factorial Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12.3](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/downloads/release/python-3123/)
[![Ubuntu 24.04](https://img.shields.io/badge/Ubuntu-24.04.5%20LTS-orange.svg)](https://releases.ubuntu.com/24.04/)
[![Engine: vLLM](https://img.shields.io/badge/Inference-vLLM-green.svg)](https://github.com/vllm-project/vllm)

Official repository for the paper:  
**"Fabricating Certainty: Analysis of Epistemic Humility through Automated Factorial Pipeline"**

---

## 📌 Overview

Large language models (LLMs) frequently struggle with **epistemic humility**—the capacity to recognize and communicate the boundaries of their knowledge. When confronted with unanswerable queries, models often fabricate certainty rather than abstaining or expressing doubt.

This repository provides an automated factorial evaluation framework designed to disentangle and measure three core capabilities:
1. **The Generator (G):** Elicits hallucinations by crafting adversarial, open-ended questions targeting knowledge boundaries.
2. **The Student (S):** Responds relying strictly on parametric memory, tested on whether it abstains or fabricates ungrounded claims.
3. **The Judge (J):** Evaluates question adherence and student hallucination/false objectivity.

   [ Hidden Ground-Truth Context (Wikipedia) ]

                     │
                     ▼
           ┌───────────────────┐
           │   Generator (G)   │  ──► Crafts category-constrained query
           └─────────┬─────────┘
                     │

   Query (Target Language, No Context) 

                     │
                     ▼
           ┌───────────────────┐
           │    Student (S)    │  ──► Answers solely from internal memory
           └─────────┬─────────┘
                     │

   Context + Query + Student Response
    
                     │
                     ▼
           ┌───────────────────┐
           │     Judge (J)     │  ──► Validates question adherence &
           └───────────────────┘      flags hallucination / false objectivity


### Key Research Findings
- **Scaling vs. Reasoning:** Increasing model scale consistently improves epistemic humility in standard instruction-tuned models. However, extended reasoning trajectories (thinking mode) do not guarantee humility and can exacerbate hallucination loops.
- **Adversarial Vulnerability:** Smaller generator models effectively induce hallucinations by generating malformed or ambiguous prompts.
- **Judge Reliability:** Large open-weight models ($\ge 27\text{B}$) achieve inter-rater agreement comparable to human annotators ($\kappa \approx 0.49$), whereas smaller judges exhibit substantial blind spots.
- **Multi-Turn Degradation:** Hallucinations compound across conversational depth. An initial hallucination at Turn 1 increases the conditional probability of subsequent fabrications at Turn 4 to over $84\%$.

---

## 📂 Epistemic Categories & Languages

The pipeline evaluates models against **3,000 balanced topics** (250 English, 250 Slovenian per category) extracted dynamically from Wikipedia:

| Category | Description | Target Adversarial Behavior |
| :--- | :--- | :--- |
| **No scientific consensus** | Open problems in science (cosmology, biology, quantum physics) | Demands a single definitive factual cause/mechanism |
| **Imagination** | Distant future projections and speculative technologies | Demands precise factual predictions/dates |
| **Completely subjective** | Art, aesthetics, culinary arts, personal opinions | Demands an objective truth or queries user personal state |
| **Too many variables** | Chaotic systems, macroeconomic projections, meteorology | Demands exact calculations for unpredictable systems |
| **Philosophical** | Moral dilemmas, metaphysics, ethical standpoints | Demands a single "correct" objective stance |
| **Question with unknown answer** | Obscure trivia, unsolved mysteries, lost historical items | Demands exact dates/names absent from parametric memory |

---

## 🛠️ Environment & Installation

### System Specifications
- **Operating System:** Ubuntu 24.04.5 LTS
- **Python Version:** `3.12.3`
- **Recommended Hardware:** NVIDIA GPU with at least 24 GB to 48 GB VRAM (for models up to 32B with 4-bit / 8-bit quantization via bitsandbytes).

### Setup Virtual Environment


#### Clone the repository


    git clone https://github.com/your-username/factorial_hallucination_analysis.git

    cd factorial_hallucination_analysis

    # Create and activate a clean Python 3.12.3 venv
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install Dependencies

    # Install the pinned dependencies from requirements.txt:

    pip install -r requirements.txt


Note: vLLM relies on CUDA 12+. Verify your CUDA toolkit installation before installing PyTorch and vLLM:


    python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"


## 🚀 Pipeline Execution

The experiment pipeline consists of data generation, model inference, cross-judge evaluation, and statistical reporting.

Step 1: Generate Datasets & Run Multi-Turn Interactions

        ├── Gemma 3 Family (4B, 12B, 27B)
        └── Qwen 3 Family (8B, 14B, 32B — not thinking & Thinking)
                          │
                          ▼
Step 2: Cross-Evaluate Large Judges (G27B, Q32B-NT, Q32B-T)

                          │
                          ▼
Step 3: Compute Inter-Rater Reliability (Fleiss' & Cohen's κ)

                          │
                          ▼
Step 4: Compute Factorial Metrics & Statistical Summaries

### 1. Generate Datasets & Execute Factorial Probing

Each script handles:

Asynchronous Wikipedia topic caching (bilingual).

Phase 1: Adversarial Question Generation ($G$).

Phase 2: Student Answering ($S$).

Phase 3: LLM-as-a-Judge Evaluation ($J$).

Phase 4: 4-Turn Conversational Degradation on the largest model tier.

#### Run Gemma 3 Experiment Suite

Executes the full $3 \times 3 \times 3$ factorial matrix for google/gemma-3-4b-it, google/gemma-3-12b-it, and google/gemma-3-27b-it:


    python create_hallucination_dataset_for_analysis_gemma.py

Outputs are saved to ./hallucination_datasets_gemma/.

#### Run Qwen 3 Experiment Suite (Non-Thinking & Thinking)

Executes the factorial pipeline for Qwen/Qwen3-8B, Qwen/Qwen3-14B, and Qwen/Qwen3-32B under both deterministic (enable_thinking=False) and reasoning-oriented (enable_thinking=True) regimes:

    python create_hallucination_dataset_for_analysis_qwen.py

Outputs are saved to ./hallucination_datasets_qwen/ and ./hallucination_datasets_qwen_thinking/.

### 2. Large Judge Cross-Evaluation Study

To validate evaluator reliability and ensure evaluations are not artifacts of a single judge's internal biases, run the cross-evaluation study across all 81,000 interactions evaluated by Gemma 27B, Qwen 32B NT, and Qwen 32B Thinking:

    python cross_eval_judges_hallucination.py

### 3. Compute Judge Agreement Metrics

Compute multi-rater agreement (Fleiss' $\kappa$) across all three judges and pairwise agreement (Cohen's $\kappa$ and Match %) broken down by language and question category:

    python get_cross_eval_judge_agreement.py

### 4. Compute Analysis & Paper Results

Generate all remaining tables, figures, and statistical breakdowns reported in the paper:

Experiment A: Student vulnerability and hallucination rates across model sizes and categories.

Experiment B: Generator Attack Success Rate (ASR) and Question Validity adherence.

Experiment C: Small vs. Large judge evaluator blind-spot rates (Match % and Cohen's $\kappa$).
Multi-Turn Analysis: Turn-by-turn hallucination degradation ($T_1 \to T_4$) and conditional error snowballing: $P(\text{Error}t \mid \text{Error}{<t})$.

    python analyse_hallucination_results.py

## 🗂️ Project Structure

    .
    ├── create_hallucination_dataset_for_analysis_gemma.py  # Gemma 3 Factorial & Multi-turn Pipeline
    ├── create_hallucination_dataset_for_analysis_qwen.py   # Qwen 3 Factorial & Multi-turn Pipeline (NT & T)
    ├── cross_eval_judges_hallucination.py                  # Large judge cross-evaluation inference
    ├── get_cross_eval_judge_agreement.py                   # Computes Fleiss' and Cohen's Kappa for judges
    ├── analyse_hallucination_results.py                    # Analyzes Exp A, Exp B, Exp C, and Multi-turn
    ├── requirements.txt                                    # Python dependencies
    ├── wiki_cache/                                         # Shared, cached Wikipedia crawls (generated)
    │   └── wiki_topics_cache.json


## 📜 Citation

If you use this benchmark, code, or findings in your research, please cite our paper:

      @article{fabricating_certainty2026,
        title={Fabricating Certainty: Analysis of Epistemic Humility through Automated Factorial Pipeline},
        author={Anonymous author},
        journal={arXiv preprint},
        year={2026}
      }

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.




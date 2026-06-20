# Biomedical LLM Fine-Tuning Benchmarks

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Transformers](https://img.shields.io/badge/Transformers-QLoRA-orange)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![TRL](https://img.shields.io/badge/TRL-SFTTrainer-purple)
![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-red)
![Task](https://img.shields.io/badge/Task-PubMedQA%20Classification-lightgrey)

## Overview

This project benchmarks two biomedical LLM fine-tuning pipelines for PubMedQA-style answer classification.

The task is a three-class biomedical question answering classification problem where the model predicts one of:

- `yes`
- `no`
- `maybe`

The repository compares two notebooks built around the same base model family, dataset split, class-balancing strategy, LoRA capacity, sequence length, optimizer, scheduler, and evaluation protocol:

1. **Standard Hugging Face QLoRA**
2. **Unsloth QLoRA**

The goal is to measure the runtime and validation behavior of both pipelines under comparable training settings, not to claim an official state-of-the-art result or leaderboard submission.

## Experiments

| Experiment | Notebook | Base Model | Framework |
|---|---|---|---|
| Standard Hugging Face QLoRA | `qwen2-5-7b-pubmed-sota-optimized.ipynb` | `Qwen/Qwen2.5-7B-Instruct` | `Transformers + BitsAndBytes + PEFT + TRL` |
| Unsloth QLoRA | `qwen2-5-7b-pubmed-sota-unsloth-finetune.ipynb` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | `Unsloth FastLanguageModel + TRL` |

Both notebooks fine-tune a 4-bit quantized Qwen2.5 7B instruction model using LoRA adapters for biomedical classification.

## Dataset

The experiments use expert-labeled PubMedQA examples with three answer labels: `yes`, `no`, and `maybe`.

| Dataset Property | Value |
|---|---:|
| Raw expert-labeled examples | `1,000` |
| Labels | `yes`, `no`, `maybe` |
| Validation split | `10% stratified` |
| Validation size | `100` |
| Final train size after balancing and dual-format prompting | `2,982` |

### Original Training Distribution

| Label | Count |
|---|---:|
| `yes` | `497` |
| `no` | `304` |
| `maybe` | `99` |

### Validation Distribution

| Label | Count |
|---|---:|
| `yes` | `55` |
| `no` | `34` |
| `maybe` | `11` |

### Balanced Training Distribution

| Label | Count |
|---|---:|
| `yes` | `497` |
| `no` | `497` |
| `maybe` | `497` |

The training set is balanced before dual-format prompting. Each balanced example is converted into two prompt styles, producing a final training size of `2,982`.

## Training Setup

| Configuration | Standard Hugging Face QLoRA | Unsloth QLoRA |
|---|---:|---:|
| Quantization | `4-bit` | `4-bit` |
| Epochs | `3` | `3` |
| Training steps | `561` | `561` |
| Batch size | `1` | `1` |
| Gradient accumulation | `16` | `16` |
| Effective batch size | `16` | `16` |
| Max sequence length | `1024` | `1024` |
| Learning rate | `1.2e-4` | `1.2e-4` |
| Optimizer | `paged_adamw_8bit` | `paged_adamw_8bit` |
| Scheduler | `cosine` | `cosine` |
| LoRA rank | `64` | `64` |
| LoRA alpha | `128` | `128` |
| LoRA dropout | `0.05` | `0.0` |
| Trainable parameters | `161,480,704` | `161,480,704` |
| Trainable percentage | `2.0764%` | `2.0764%` |
| Best checkpoint | `./results_qwen_pubmed_advanced/checkpoint-187` | `./results_qwen_pubmed_unsloth/checkpoint-187` |

The two experiments use the same number of steps and the same trainable parameter count. The main implementation difference is the model loading and training optimization path.

## Runtime Comparison

| Runtime Metric | Standard Hugging Face QLoRA | Unsloth QLoRA |
|---|---:|---:|
| Train time | `5h 18m 17s` | `4h 09m 21s` |
| Train time in seconds | `19,097` | `14,961` |
| Eval runtime | `73.7815s` | `58.4805s` |
| Trainer train runtime | Not logged in summary | `14560.6775s` |
| Wall-clock train runtime logged by notebook | Not logged in summary | `14563.0088s` |
| Train samples per second | Not logged in summary | `0.614` |
| Train steps per second | Not logged in summary | `0.039` |
| Peak reserved GPU memory | Not logged in summary | `7.8848 GB` |

Runtime delta:

```text
Standard seconds = 19,097
Unsloth seconds = 14,961
Time saved = 19,097 - 14,961 = 4,136 seconds = 1h 08m 56s
Training time reduction = 4,136 / 19,097 = 21.66%
Speedup = 19,097 / 14,961 = 1.28x faster
```

Unsloth completed the same 561-step training run `1h 08m 56s` faster than the standard Hugging Face QLoRA notebook, corresponding to a `21.66%` training time reduction and a `1.28x` speedup.

## Validation Results

| Metric | Standard Hugging Face QLoRA | Unsloth QLoRA |
|---|---:|---:|
| Train loss | `0.4705` | `0.4675` |
| Eval loss | `1.6900` | `1.7482` |
| Accuracy | `79.00%` | `78.00%` |
| Macro F1 | `0.5720` | `0.5498` |
| Weighted F1 | `0.7597` | `0.7346` |

The standard Hugging Face QLoRA run achieved slightly higher validation accuracy, macro F1, and weighted F1. The Unsloth run achieved very similar train loss and substantially lower runtime, but its validation metrics were marginally lower.

## Per-Class Evaluation

### Standard Hugging Face QLoRA

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `yes` | `0.82` | `0.85` | `0.84` | `55` |
| `no` | `0.82` | `0.94` | `0.88` | `34` |
| `maybe` | `0.00` | `0.00` | `0.00` | `11` |
| `macro avg` | `0.55` | `0.60` | `0.57` | `100` |
| `weighted avg` | `0.73` | `0.79` | `0.76` | `100` |

### Unsloth QLoRA

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `yes` | `0.79` | `0.87` | `0.83` | `55` |
| `no` | `0.77` | `0.88` | `0.82` | `34` |
| `maybe` | `0.00` | `0.00` | `0.00` | `11` |
| `macro avg` | `0.52` | `0.59` | `0.55` | `100` |
| `weighted avg` | `0.69` | `0.78` | `0.73` | `100` |

### Direct Per-Class Comparison

| Class | Metric | Standard Hugging Face QLoRA | Unsloth QLoRA |
|---|---|---:|---:|
| `yes` | Precision | `0.82` | `0.79` |
| `yes` | Recall | `0.85` | `0.87` |
| `yes` | F1 | `0.84` | `0.83` |
| `no` | Precision | `0.82` | `0.77` |
| `no` | Recall | `0.94` | `0.88` |
| `no` | F1 | `0.88` | `0.82` |
| `maybe` | Precision | `0.00` | `0.00` |
| `maybe` | Recall | `0.00` | `0.00` |
| `maybe` | F1 | `0.00` | `0.00` |

Both models performed well on the majority `yes` and `no` classes but failed to recover the minority `maybe` class on the validation split.

## Error Analysis

The main failure mode in both experiments is the complete absence of correct `maybe` predictions.

| Observation | Interpretation |
|---|---|
| `maybe` F1 is `0.00` in both runs | The model does not learn a reliable decision boundary for uncertain biomedical answers. |
| `maybe` validation support is only `11` examples | Evaluation for this class is highly sensitive to a small number of predictions. |
| Training was balanced, but validation remained naturally imbalanced | Balancing improved exposure during training but did not guarantee minority-class generalization. |
| `yes` and `no` dominate validation performance | Weighted F1 remains much higher than macro F1 because the larger classes perform substantially better. |
| Eval loss is high relative to train loss | The model may be overfitting the training prompts or becoming overconfident on validation examples. |

The results suggest that the model learns the dominant biomedical entailment-style labels more effectively than the uncertainty label. The `maybe` class likely requires additional data, better uncertainty-specific prompting, calibrated decoding, or a classifier-style objective rather than pure generative supervision.

## Why Unsloth Was Faster

The Unsloth notebook replaces the standard Hugging Face model preparation path with Unsloth’s optimized `FastLanguageModel` path.

Unsloth log details:

```text
Unsloth patched 28 layers with 28 QKV layers, 28 O layers and 28 MLP layers
Num GPUs used = 1
Data Parallel GPUs = 1
Unsloth: Will smartly offload gradients to save VRAM
Unsloth: Double buffering enabled for backward pass
```

The speedup is mainly explained by the following implementation differences:

| Area | Standard Hugging Face QLoRA | Unsloth QLoRA |
|---|---|---|
| Model loading | `AutoModelForCausalLM` with BitsAndBytes quantization | Pre-quantized Unsloth 4-bit checkpoint |
| LoRA preparation | `prepare_model_for_kbit_training` + PEFT | `FastLanguageModel.get_peft_model` |
| Layer patching | Standard module execution path | Patched QKV, output, and MLP layers |
| Gradient checkpointing | Standard gradient checkpointing | Unsloth gradient checkpointing |
| Memory behavior | Standard QLoRA memory flow | Gradient offloading and double buffering |
| Runtime result | `19,097s` | `14,961s` |

The Unsloth run reduces training time while keeping the same number of steps, effective batch size, sequence length, LoRA rank, LoRA alpha, and trainable parameter count.

## Benchmark Context

This benchmark is a controlled notebook-level comparison, not an official PubMedQA benchmark submission.

Important context:

- Both notebooks use QLoRA fine-tuning on Qwen2.5 7B instruction models.
- Both notebooks use the same validation size of `100` examples.
- Both notebooks train for `3` epochs and `561` steps.
- Both notebooks use deterministic classification evaluation over the validation set.
- The results are single-run measurements and may vary across GPU runtime, package versions, CUDA setup, and random seed.
- The term `sota` appears in notebook filenames and artifact names, but these experiments should not be interpreted as official state-of-the-art results.

## Generated Artifacts

### Standard Hugging Face QLoRA

| Artifact | Path |
|---|---|
| Best checkpoint | `./results_qwen_pubmed_advanced/checkpoint-187` |
| Saved adapter/model directory | `./results_qwen_pubmed_advanced/Qwen2.5-7B-PubMedQA-SOTA` |
| Run summary | `./results_qwen_pubmed_advanced/run_summary.csv` |
| Confusion matrix | `./results_qwen_pubmed_advanced/confusion_matrix.png` |
| Per-class metrics plot | `./results_qwen_pubmed_advanced/per_class_metrics.png` |
| Loss curves | `./results_qwen_pubmed_advanced/loss_curves.png` |
| Validation predictions | `./results_qwen_pubmed_advanced/validation_predictions.csv` |

### Unsloth QLoRA

| Artifact | Path |
|---|---|
| Best checkpoint | `./results_qwen_pubmed_unsloth/checkpoint-187` |
| Saved LoRA adapter directory | `./results_qwen_pubmed_unsloth/Qwen2.5-7B-PubMedQA-Unsloth-QLoRA` |
| Run summary | `./results_qwen_pubmed_unsloth/run_summary.csv` |
| Confusion matrix | `./results_qwen_pubmed_unsloth/confusion_matrix.png` |
| Per-class metrics plot | `./results_qwen_pubmed_unsloth/per_class_metrics.png` |
| Loss curves | `./results_qwen_pubmed_unsloth/loss_curves.png` |
| Validation predictions | `./results_qwen_pubmed_unsloth/validation_predictions.csv` |

## Reproducibility Notes

To reproduce the comparison, both notebooks should be run with the same dataset, split, seed, GPU environment, and package versions.

Recommended reproducibility checks:

1. Use the same raw PubMedQA expert-labeled examples.
2. Preserve the `10%` stratified validation split.
3. Keep the same label order: `yes`, `no`, `maybe`.
4. Keep the same balancing logic before dual-format prompting.
5. Keep the same training hyperparameters.
6. Compare runs using the same hardware and CUDA runtime.
7. Save row-level validation predictions for error inspection.
8. Compare both aggregate metrics and per-class metrics.

The Unsloth notebook detected two Tesla T4 GPUs in the runtime, while the Unsloth trainer log reported:

```text
Num GPUs used = 1
Data Parallel GPUs = 1
```

For strict runtime comparisons, GPU allocation and device usage should be verified before each run.

## Limitations

- The validation set contains only `100` examples.
- The `maybe` class has only `11` validation examples.
- Both runs failed to correctly recover the `maybe` class.
- The benchmark uses a single train/validation split.
- The experiments report single-run results without variance across multiple seeds.
- The standard and Unsloth notebooks differ in LoRA dropout: `0.05` for standard QLoRA and `0.0` for Unsloth QLoRA.
- Runtime measurements may vary depending on GPU availability, thermal behavior, CUDA version, package versions, and notebook environment.
- The task is evaluated as label classification, while the model is trained through generative supervised fine-tuning.
- The benchmark is not an official PubMedQA leaderboard submission.

## Future Work

Potential improvements:

1. Run multiple seeds and report mean and standard deviation.
2. Use cross-validation or repeated stratified splits.
3. Increase the number of expert-labeled biomedical examples.
4. Improve the `maybe` class with targeted augmentation or uncertainty-focused examples.
5. Test classifier-head fine-tuning against generative label prediction.
6. Evaluate calibration and confidence scores.
7. Compare LoRA dropout settings under identical configurations.
8. Add confusion matrices and row-level error reports to the README.
9. Evaluate larger or newer biomedical-capable instruction models.
10. Test inference-time prompt variants for more stable `maybe` predictions.
11. Compare adapter-only saving, merged model saving, and deployment formats.
12. Track GPU memory, throughput, and wall-clock time consistently for both notebooks.

## References

- [PubMedQA Dataset](https://pubmedqa.github.io/)
- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Unsloth](https://github.com/unslothai/unsloth)
- [unsloth/Qwen2.5-7B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct-bnb-4bit)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [TRL](https://github.com/huggingface/trl)
- [BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

## Disclaimer

This repository documents experimental biomedical LLM fine-tuning runs for technical comparison purposes only.

The results are not medical advice, are not intended for clinical use, and should not be used for biomedical decision-making without rigorous validation. The reported metrics are based on a small validation split and should be interpreted as notebook-level experimental results rather than official benchmark or leaderboard performance.

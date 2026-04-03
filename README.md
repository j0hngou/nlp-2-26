# NLP2 Project B: Test-Time Scaling for Multimodal Reasoning

Starter repository for the NLP2 course project on test-time scaling (TTS) strategies
for vision-language models, using the [ImageCLEF 2026 MultimodalReasoning](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning) shared task.

## Quick Start

```bash
pip install -r requirements.txt
# Then open starter.ipynb in Jupyter/Colab, or run:
python starter.py
```

On **Google Colab**, open `starter.ipynb` and the first cell installs everything.

## Repository Structure

```
nlp2-tts-vlm/
├── README.md
├── requirements.txt
├── gpu_server.sh              # SSH tunnel for remote GPU access
├── starter.ipynb              # Main starter notebook
├── starter.py                 # Same content as .py script
├── eval/
│   └── evaluate.py            # Stratified accuracy evaluation (development)
└── configs/
    └── default.yaml           # Model/generation hyperparameters
```

## What the Starter Provides

The notebook has 7 sections:

1. **Setup**: imports, GPU check, seed
2. **Data loading**: EXAMS-V (MCQ) and OpenQA datasets from HuggingFace
3. **Model loading**: Qwen2.5-VL-7B-Instruct with 4-bit quantization
4. **Zero-shot baseline**: image-to-answer prompting
5. **Chain-of-thought**: step-by-step reasoning before answering
6. **Self-consistency**: majority voting over N sampled chains
7. **Evaluation**: accuracy (overall + stratified), competition-format export, scaling curves

## What You Need to Implement

- [ ] At least one **search strategy** (Tree-of-Thought, beam search, simplified MCTS)
- [ ] At least one **verification strategy** (agentic verification, VisualPRM, LLM-as-judge)
- [ ] **Stratified analysis** by subject, language, and visual content type
- [ ] **Compute scaling curves** (accuracy vs. number of VLM calls)
- [ ] Process **test data** when released for competition submission

Skeleton functions with docstrings are in Cell 6 of the notebook:
`tree_of_thought()`, `verify_chain()`, `best_of_n_with_verification()`.

## Evaluation

For quick dev iteration, use the included eval script:

```bash
python eval/evaluate.py --predictions preds.json --references refs.json --stratify-by subject language
```

For official competition evaluation, use the scripts from the [ImageCLEF repo](https://github.com/mbzuai-nlp/ImageCLEF-MultimodalReasoning):

```bash
# MCQ (accuracy)
python src/evaluation/evaluate_mcq.py --pred_file pred.json --gold_file gold.json --print_score True

# OpenQA (BLEU, ROUGE, METEOR, COMET)
python src/evaluation/evaluate_qa.py --pred_file pred.json --gold_file gold.json
```

Submit via [AI4Media-Bench](https://ai4media-bench.aimultimedialab.ro/) (max 20 submissions/day, 200 total).

## Choosing a Model

The notebook uses **Qwen2.5-VL-7B-Instruct** as a starting point, but this is probably not
the best model for the task. You should explore alternatives and figure out what works.

Places to look:

- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): compare VLMs across benchmarks. Find the benchmarks closest to our setting (multilingual, OCR-heavy, science/math reasoning with images) and see what ranks well there.
- [LMSYS Chatbot Arena](https://lmarena.ai/): general LLM/VLM Elo ratings.

These aren't the only resources. Look for other benchmarks and arenas that match our task.

Some models to start with:

| Model | Params | Notes |
|-------|--------|-------|
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 7B | Default in the notebook |
| [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3B | Faster iteration |
| [OpenGVLab/VisualPRM-8B](https://huggingface.co/OpenGVLab/VisualPRM-8B) | 8B | Process reward model for verification |

### Quantization

Quantization lets you fit bigger models on the same GPU. A quantized 14B model might beat
a full-precision 7B one, so try different combos and measure. See
[A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)
for background.

Options:

- **4-bit** (NF4 via BitsAndBytes): ~4x memory savings. Default in the notebook. 7B fits in ~6 GB.
- **8-bit** (INT8): ~2x memory savings, closer to full-precision quality.
- **AWQ / GPTQ**: pre-quantized weights on HuggingFace. Often faster than BitsAndBytes.

On a 24 GB GPU with 4-bit you can fit up to ~30B parameters.

## Inference with vLLM

The notebook uses plain PyTorch inference (one image at a time), which is slow. For actual
experiments, use [vLLM](https://docs.vllm.ai/) instead. It serves the model behind an
OpenAI-compatible API and is much faster.

Start the server:
```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt image=1
```

`--max-model-len` must be large enough to fit image tokens + text. Qwen2.5-VL can expand a
single image to 10k-16k tokens depending on resolution.

Then use the OpenAI client to query it (see the vLLM cell in the notebook). Since vLLM handles
batching internally, you can send multiple requests concurrently (e.g., with `asyncio` or
`concurrent.futures.ThreadPoolExecutor`) to get much higher throughput than sequential calls.
This is especially useful for self-consistency and scaling curves where you need many
independent generations per image. Check the [vLLM docs](https://docs.vllm.ai/) for tuning.

## Compute Constraints

- Competition requires **single A40 GPU**, open-weights models
- Model tracks: **<=7B** (tiny) or **>=8B** (normal)

## Data Sources

- **EXAMS-V** (MCQ): [`MBZUAI/EXAMS-V`](https://huggingface.co/datasets/MBZUAI/EXAMS-V), ~20k questions, 13 languages, 20 subjects
- **OpenQA-Textual**: [`SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual`](https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Textual), 625 samples
- **OpenQA-Visual**: [`SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual`](https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual), 768 samples

## References

- [ImageCLEF 2025 MSA (1st place)](https://arxiv.org/abs/2507.11114), describe-then-reason pipeline
- [ImageCLEF 2025 ContextDrift (2nd)](https://ceur-ws.org/Vol-4038/paper_194.pdf), thinking budget analysis
- [ImageCLEF 2025 Task Overview](https://ceur-ws.org/Vol-4038/paper_174.pdf)

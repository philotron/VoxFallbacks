# Not All Fallbacks Are Failures: Understanding and Recovering from Fallbacks in Mobile Voice Assistants

Data, code, and prediction outputs accompanying our **EMNLP 2026 Industry Track** paper:
[Not All Fallbacks Are Failures: Understanding and Recovering from Fallbacks in Mobile Voice Assistants](https://arxiv.org/abs/2608.30738).

**Phillip Schneider, Alexandre Mercier, Joshua Oehms, Kristiina Jokinen, and Florian Matthes**

We study fallback handling in a deployed smartwatch-based voice assistant. This repository provides **VoxFallbacks**, an annotated dataset of 3,030 naturally occurring German fallback utterances, together with experiments comparing embedding-based classifiers and language models for detecting intended speech and routing recoverable requests.

## Study Citation

Please cite the paper when using the dataset or experiments. The following entry references the arXiv version; machine-readable citation metadata is also provided in [CITATION.cff](CITATION.cff).

```bibtex
@misc{schneider2026fallbacks,
	title = {Not All Fallbacks Are Failures: Understanding and Recovering from Fallbacks in Mobile Voice Assistants},
	author = {Schneider, Phillip and Mercier, Alexandre and Oehms, Joshua and Jokinen, Kristiina and Matthes, Florian},
	year = {2026},
	eprint = {2608.30738},
	archivePrefix = {arXiv},
	primaryClass = {cs.CL},
	doi = {10.48550/arXiv.2608.30738},
	url = {https://arxiv.org/abs/2608.30738}
}
```

## License

The author-owned contents of this repository, including the dataset, code, documentation, and prediction outputs, are licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)**. See the [license summary](https://creativecommons.org/licenses/by-nc-nd/4.0/) and the full [LICENSE](LICENSE).

The license permits sharing unadapted material for noncommercial purposes with attribution. Adapted material may be produced and reproduced for noncommercial purposes, but may not be shared under this license. Consult the full license for its terms and applicable exceptions. The linked paper, external dependencies, and pretrained models are governed separately by their respective terms.

## General Information

### Dataset

VoxFallbacks consists of single-turn transcripts collected over six months from a German-speaking smartwatch voice assistant used by more than 500 people. The paper describes the removal of empty inputs and exact duplicates, manual replacement of identifying information, and randomization of utterance order. Audio recordings and complete dialogue histories are not included. Because the corpus is deduplicated, its class distribution should not be interpreted as raw production frequency.

| File | Contents |
| --- | --- |
| [data/dataset.xlsx](data/dataset.xlsx) | 3,030 utterances and final annotations on the `Data` sheet. |
| [data/dataset_cv.xlsx](data/dataset_cv.xlsx) | The same rows, with the published stratified five-fold assignment on the `Data` sheet. |
| [data/dataset_inter_annotator.xlsx](data/dataset_inter_annotator.xlsx) | Annotation workbook containing the full corpus and individual ratings on `Sheet1`, including 350 utterances jointly rated by three annotators. |

The main dataset columns are:

| Column | Description |
| --- | --- |
| `id` | Unique utterance identifier used to join data and predictions. |
| `resolved_query` | Anonymized German utterance transcript. |
| `label` | Final hierarchical annotation. |
| `cv_fold` | Fold identifier from 0 to 4, present only in the CV workbook; each fold contains 606 utterances. |

The taxonomy distinguishes unintended input, intended input with unclear intent, and intended input with a clear request for an available or unavailable skill. There are 22 observed raw labels: 18 available skills, two unclear-intent categories (`unspecific` and `multiple`), one unavailable-skill category, and `unintended`.

The experiments use two tasks:

- **Binary classification:** all 3,030 utterances, with `unintended` mapped to 0 (2,138 examples) and all intended labels mapped to 1 (892 examples).
- **Skill routing:** the 892 intended utterances, mapped to 18 observed skill labels plus `out_of_scope` and `unclear`. The two unclear-intent categories are merged. The code's label mapping supports additional skills that are not observed in this dataset.

### Repository Contents

```text
.
|-- README.MD
|-- LICENSE
|-- CITATION.cff
|-- .gitignore
|-- requirements.txt
|-- env.template
|-- data/
|   |-- dataset.xlsx
|   |-- dataset_cv.xlsx
|   `-- dataset_inter_annotator.xlsx
|-- src/
|   |-- experiments.ipynb
|   |-- train_classifier.py
|   |-- predict.py
|   `-- utils.py
`-- results/
		`-- predictions/
```

[src/experiments.ipynb](src/experiments.ipynb) contains the experiment settings and prompts. [src/train_classifier.py](src/train_classifier.py) implements the embedding-based classifiers, [src/predict.py](src/predict.py) implements language-model evaluation and optional Rasa conditioning, and [src/utils.py](src/utils.py) provides dataset loading, label mappings, and evaluation helpers.

### Setup

Use **Python 3.13** and run the following commands from the repository root. Direct dependencies are pinned in [requirements.txt](requirements.txt); this is not a complete lockfile of transitive dependencies.

On Linux or macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, with `python` pointing to Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start Jupyter **from the source directory** so that relative paths and sibling imports resolve:

```bash
cd src
python -m jupyter notebook experiments.ipynb
```

In an IDE, select the virtual environment as the notebook kernel and use `src` as the kernel's working directory. Run the relevant sections rather than executing the entire notebook indiscriminately: later cells train models, call external services, and overwrite their corresponding prediction CSVs.

### Evaluate Included Predictions

Run only the **Evaluate Included Predictions** section near the beginning of the notebook. It uses the existing evaluation helpers and published fold assignments to evaluate every included prediction CSV. This does not download model weights, retrain classifiers, or require an API key, Ollama, or Rasa.

The **Inter Annotator Agreement** section can also be run without model services. It computes agreement from the three annotator columns and uses 10,000 bootstrap resamples.

### Reproduce Embedding Experiments

Run **Prerequisites**, followed by the desired **Baseline (Encoder + Trained Classifier)** cells. The notebook evaluates logistic regression, SVM, and random forest for the binary task, and logistic regression for skill routing, using:

- `intfloat/multilingual-e5-large-instruct`
- `BAAI/bge-m3`

The first run downloads encoder weights. Cached encoder files are stored under `models/embeddings_cache` at the repository root and are not included in the release. Internet access and sufficient memory are required; training and inference depend on the available hardware. The encoder loader enables `trust_remote_code`, so review the selected model's code before running it.

Keep the shipped CV workbook to use the published folds. If it is missing, **Prerequisites** regenerates it using stratified five-fold splitting with seed 42. The release does not bundle trained classifier weights.

### Run Language-Model Experiments

After **Prerequisites**, run the prompt cells and the model-specific cells for the desired task. Binary experiments use zero-shot and few-shot prompts; routing experiments compare the default prompt with Rasa-conditioned prompting.

**Local models.** The notebook uses `gemma4:e4b-mlx` and `qwen3.5:4b-mlx` through local Ollama-compatible interfaces. These are the original local model identifiers, not guaranteed publicly available Ollama pull tags. Binary evaluation uses the Ollama client and token log probabilities; routing uses the HTTP chat endpoint at `http://localhost:11434/api/chat`. A compatible serving setup, model files, and response format are required. The original experiments used an Apple M3 Pro MacBook Pro with 18 GB unified memory; the MLX setup is not a turnkey Windows deployment. The model weights and original serving configuration are not bundled.

**OpenAI.** GPT experiments use `gpt-4.1-nano-2025-04-14`. Copy [env.template](env.template) to `.env` in the repository root and replace the placeholder with your own `OPENAI_API_KEY`. For example, from the root:

```bash
cp env.template .env
```

Or in PowerShell:

```powershell
Copy-Item env.template .env
```

Never commit the populated file. An API key is needed only for GPT calls, not for local imports or included-results evaluation. Requests transmit utterance text to the provider, require network access, and incur API costs. Service availability and outputs may change over time.

**Rasa-conditioned routing.** Cells with `prompt_variant="rasa"` require a separately supplied trained Rasa model and custom ranking service. The code posts to `http://localhost:5005/webhooks/rest/webhook` and expects response text containing a Python-literal list of dictionaries with intent `name` fields, in ranked order. A standard Rasa REST response alone is insufficient. The model, training phrases, and custom service are not included, so these variants cannot be rerun from this repository alone. Their released predictions can still be evaluated without the service.

### Results and Reproduction Limits

[results/predictions](results/predictions) contains 13 binary and eight skill-routing CSVs. Twelve binary files correspond to the main encoder/classifier and zero-/few-shot LLM configurations. The additional [binary_logistic_regression_e5_c1.csv](results/predictions/binary_logistic_regression_e5_c1.csv) is retained as a supplementary variant. Routing files cover two embedding baselines and three LLMs with default and Rasa-conditioned prompts.

Each CSV is indexed by utterance ID and includes `prediction`, `label`, and `latency` in seconds. Binary files also contain `score`; routing outputs may include a confidence score or an `invalid_prediction` field. Filenames identify the task, model, and prompt variant. Binary files cover all 3,030 utterances; routing files cover the 892 intended utterances.

The evaluation helpers report means and standard deviations across the five folds. Binary metrics include accuracy, precision, recall, F1, ROC AUC, and average precision; routing uses macro-averaged precision, recall, and F1 alongside accuracy. Reported classifier latency excludes embedding computation. Local serving, hardware, retrieval, and API/network overhead affect other timings, so latency is not directly portable between machines.

The included outputs support re-evaluation without model services. Full reruns additionally depend on encoder downloads, the original local serving setup, OpenAI access, or the external Rasa artifacts, depending on the variant. These experiments evaluate fallback classification and routing, not end-to-end recovery success or a systematic evaluation of clarification generation.

### Acknowledgments

We thank Lutz Frick, founder of ALMA PHIL, for his support of this work.

### Questions

For questions about the dataset or study, contact [Phillip Schneider](mailto:phillip.schneider@tum.de).
# 🩺 MedNoteClassifier Workspace

> **Clinical Feature Span Extraction Suite & Interactive Web Application**

[![Node.js](https://img.shields.io/badge/Node.js-v18%2B%20%7C%20v20%2B-brightgreen?logo=node.js)](https://nodejs.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF?logo=vite)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-06B6D4?logo=tailwindcss)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](MedNoteClassifier/LICENSE)

---

## 📌 Overview

**MedNoteClassifier** is a full-stack clinical NLP solution designed for automated span extraction on the **NBME® — Score Clinical Patient Notes** dataset (USMLE® Step 2 Clinical Skills). The system identifies exact character offsets `[start, end]` in unstructured patient notes corresponding to standardized rubric concepts (e.g., *"heart pounding"*, *"family history of MI"*).

The workspace bundles a high-performance **Python ML Package** ([`MedNoteClassifier/`](file:///c:/Users/athar/Desktop/Projects/MedNoteClassifier/MedNoteClassifier/README.md)) with a modern **React 19 & Vite Web Application** for interactive annotation, real-time prediction visualization, and model performance benchmarking.

---

## 📐 Architecture & Workflow

### 1. System Architecture

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1E293B', 'primaryTextColor': '#F8FAFC', 'primaryBorderColor': '#3B82F6', 'lineColor': '#64748B', 'secondaryColor': '#0F172A', 'tertiaryColor': '#1E293B'}}}%%
flowchart TB
    subgraph Client ["💻 Web Application Client (React 19 & Tailwind CSS v4)"]
        direction LR
        UI["🎨 Interactive Dashboard<br/><i>(TanStack Router)</i>"]
        Annotator["🔍 Clinical Span Annotator<br/><i>(Real-Time Highlighting)</i>"]
        Benchmarker["📊 Model Comparison<br/><i>(Micro-F1 Benchmark)</i>"]
    end

    subgraph AppServer ["⚡ API & Backend Service (Node.js & Express)"]
        API["📡 HTTP REST API Endpoint"]
        Bridge["⚙️ Process Executor & Manager"]
    end

    subgraph MLEngine ["🧠 Python Clinical NLP Suite (MedNoteClassifier)"]
        Preprocess["🔤 Tokenizer & Offset Indexer"]
        
        subgraph Ensemble ["🤖 Model Inference Engine"]
            Lexicon["⚡ Exact Lexicon Matcher"]
            Hybrid["🔠 Hybrid Char N-Gram LogReg"]
            Transformer["🔥 PyTorch BIO Transformer"]
            DistilBERT["🩺 Fine-Tuned PubMedBERT/DistilBERT"]
        end

        Eval["📈 Evaluation Engine<br/><i>(Character-level Micro-F1)</i>"]
    end

    subgraph DataStore ["💾 Data & Storage Layer"]
        NBME[("📋 NBME Patient Notes<br/>& Rubric Features")]
        Checkpoints[("📦 Model Checkpoints<br/>& Evaluation Metrics")]
    end

    UI -->|REST / JSON| API
    Annotator -->|Span Payload| API
    Benchmarker -->|Benchmark Query| API
    
    API --> Bridge
    Bridge -->|Subprocess Execution| MLEngine
    
    NBME --> Preprocess
    Preprocess --> Ensemble
    Ensemble --> Eval
    Eval --> Checkpoints
```

---

### 2. Clinical Span Extraction Pipeline

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1E293B', 'primaryTextColor': '#F8FAFC', 'primaryBorderColor': '#10B981', 'lineColor': '#64748B', 'secondaryColor': '#0F172A', 'tertiaryColor': '#1E293B'}}}%%
flowchart LR
    Input["📄 Unstructured Clinical Note<br/>+ Feature Rubric"] --> Tokenizer["✂️ Tokenization &<br/>Character Mapping"]
    
    subgraph Inference ["🔮 Multi-Model Inference Engine"]
        direction TB
        M1["⚡ Lexicon Matcher<br/><i>(Exact Phrase Boundaries)</i>"]
        M2["🔠 Logistic Regression<br/><i>(Char N-Gram Candidates)</i>"]
        M3["🔥 BIO Transformer<br/><i>(Sequence Token Tagging)</i>"]
    end

    Tokenizer --> M1 & M2 & M3
    
    M1 & M2 & M3 --> Decoder["📍 Character Offset Decoder<br/><code>[start, end]</code>"]
    Decoder --> Output["📊 Real-Time Highlighting<br/>& Micro-F1 Metrics"]
```

---

## 🗂 Workspace Structure

```
.
├── MedNoteClassifier/       # Python Machine Learning package (NLP Models, Datasets & Evaluation)
│   ├── app/                 # Streamlit standalone demo app
│   ├── data/                # Dataset directory (raw Kaggle CSVs & processed files)
│   ├── models/              # Saved model checkpoints (Lexicon, Hybrid LogReg, PyTorch Transformer)
│   ├── reports/             # EDA charts, evaluation metrics, and error analysis JSONs
│   ├── src/                 # ML source modules (pipeline, preprocessing, model architectures)
│   ├── tests/               # Pytest suite for ML modules
│   └── README.md            # Comprehensive ML package documentation
├── src/                     # React 19 & Vite web application frontend
├── server/                  # Backend application services & API routes
├── scripts/                 # App utility & environment startup scripts
├── startup.sh               # Unix/Linux background startup script
├── startup.bat              # Windows Command Prompt background startup script
├── startup.ps1              # Windows PowerShell background startup script
├── package.json             # Web application Node.js dependencies & scripts
└── README.md                # Root project documentation
```

---

## 🛠 Sub-Projects & Components

### 1. Clinical NLP Machine Learning Module ([`MedNoteClassifier/`](file:///c:/Users/athar/Desktop/Projects/MedNoteClassifier/MedNoteClassifier/README.md))
Located in [`MedNoteClassifier/`](file:///c:/Users/athar/Desktop/Projects/MedNoteClassifier/MedNoteClassifier/README.md).
- **Core Task:** Extracts character-level clinical rubric features from USMLE-style patient notes.
- **Model Implementations:**
  - **Exact Lexicon Matcher:** Fast exact phrase match with strict word-boundary enforcement.
  - **Hybrid Logistic Regression:** `HashingVectorizer` character n-gram classifier combined with phrase matching.
  - **PyTorch BIO Transformer:** Custom 2-layer TransformerEncoder tagging sequence tokens with BIO labels.
  - **DistilBERT / PubMedBERT Hook:** Extensible trainer for fine-tuning transformer models on medical text.
- For complete setup instructions and model benchmarks, see [`MedNoteClassifier/README.md`](file:///c:/Users/athar/Desktop/Projects/MedNoteClassifier/MedNoteClassifier/README.md).

### 2. Live Web Dashboard Application ([`src/`](file:///c:/Users/athar/Desktop/Projects/MedNoteClassifier/src))
Powered by React 19, Vite, TanStack Router/Start, and Tailwind CSS v4.
- Provides interactive visual clinical note annotation, real-time span highlighting, and model prediction comparison.

---

## 🚀 Getting Started

### Prerequisites

| Tech | Version Required | Purpose |
| --- | --- | --- |
| **Node.js** | `v18+` or `v20+` | Web dashboard & API services |
| **Python** | `v3.10+` | NLP ML models & pipeline |

---

### Quickstart Guide

#### 1. Web Application (React + Node.js)

Install Node.js dependencies and launch the dev server:

```bash
# Install NPM dependencies
npm install

# Start dev server on http://localhost:8080
npm run dev
```

#### 2. Machine Learning Pipeline (Python)

Set up a Python virtual environment and run the pipeline:

```powershell
# 1. Navigate to the ML project directory
cd MedNoteClassifier

# 2. Re-create virtual environment if migrating directories (optional)
# Remove-Item -Recurse -Force .venv

# 3. Create virtual environment
python -m venv .venv

# 4. Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# 5. Install dependencies
python -m pip install -r requirements.txt

# 6. Run full ML pipeline (EDA, Training, Evaluation)
python -m src.run_pipeline

# 7. (Optional) Run Python test suite
python -m pytest tests -q

# 8. (Optional) Run interactive Streamlit app
python -m streamlit run app/predict.py
```

---

## ⚡ Convenience Startup Scripts

To launch the web preview server automatically:

- **Linux / macOS:** `./startup.sh`
- **Windows Command Prompt:** `startup.bat`
- **Windows PowerShell:** `.\startup.ps1`

> [!NOTE]
> These scripts verify whether port `8080` is active and start `npm run dev` in the background if needed.

---

## 📜 Available NPM Scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Launch Vite dev server on port `8080` |
| `npm run build` | Build production bundle and run database migrations |
| `npm run typecheck` | Run TypeScript type checking |
| `npm run test` | Run application unit and integration tests |
| `npm run lint` | Run ESLint across codebase |
| `npm run format` | Format files with Prettier |

---

## 🔄 Continuous Integration (CI)

GitHub Actions workflows are configured under `.github/workflows/ci.yml` for automated testing and validation on `push` and `pull_request`.

---

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](file:///c:/Users/athar/Desktop/Projects/MedNoteClassifier/MedNoteClassifier/LICENSE) inside `MedNoteClassifier/` for details.

# OncoLens

OncoLens is a high-performance visual analytics system designed for multi-dimensional brain tumor characterization and genomic dysregulation mapping. Differentiating between distinct clinical pathologies is a fundamental challenge in computational biology and precision oncology. Using transcriptomic profiles, OncoLens compresses thousands of genetic dimensions into a highly interactive visual application that maps molecular and structural boundaries between healthy tissue and four aggressive forms of brain cancer: Ependymoma, Glioblastoma, Medulloblastoma, and Pilocytic Astrocytoma.

## Project Structure

```text
OncoLens/
│
├── data/
│   ├── raw/                      # Raw datasets (e.g., GSE50161 CSV, GPL570 annotation)
│   └── processed/                # Pipeline outputs (filtered, annotated, ranked datasets)
│
├── notebooks/
│   └── EDA.ipynb                 # Phase 1 Exploratory Data Analysis notebook
│
├── src/                          # Core backend processing modules
│   ├── __init__.py               # Source package initialization
│   ├── config.py                 # Global path constants and pipeline hyperparameters
│   ├── preprocessing.py          # Variance filtering and top-gene feature selection
│   ├── annotation.py             # GPL570 probe cross-referencing and cytoband mapping
│   ├── dysregulation.py          # Patient-level Dysregulation Index (DI) computation
│   ├── preprocess_network.py     # Pairwise Pearson co-expression edge list builder
│   ├── evaluate_pairs.py         # Grid-search Silhouette scorer for all gene pairs
│   └── utils.py                  # Reusable helper functions and statistical routines
│
├── app/                          # Dash web application
│   ├── __init__.py               # App package initialization
│   ├── app.py                    # Server entry point — loads data and registers callbacks
│   ├── config.py                 # Simulator gene defaults and temperature parameter
│   ├── callbacks.py              # Reactive callback handlers for all dashboard panels
│   └── layouts/                  # Modular UI layout components
│       ├── __init__.py           # Layout package exports
│       ├── theme.py              # Design tokens (colors, plot templates)
│       ├── dashboard.py          # Top-level dashboard grid assembly
│       ├── header.py             # Dashboard header bar
│       ├── contour_card.py       # Joint gene phenotyping contour/scatter card
│       ├── hotspot_card.py       # Chromosomal hotspot visualization card
│       ├── network_card.py       # Co-expression network graph card
│       ├── expression_card.py    # Gene expression profile explorer card
│       ├── viz_card.py           # Generic visualization card wrapper
│       ├── simulator_card.py     # Virtual Expression Assayer (simulator) card
│       └── summary_cards.py      # Top-level KPI summary statistics cards
│
├── assets/                       # CSS stylesheets, static images, and custom JS
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Ignored files (cache, large binaries, secrets)
```

## Dataset

The primary expression dataset is **GSE50161** (Brain tumor transcriptomics, Affymetrix HG-U133 Plus 2.0 platform).

- Download `Brain_GSE50161.csv` and place it under `data/raw/`.
- The GPL570 platform annotation file (`GPL570.annot.gz`) is automatically downloaded by `src/annotation.py` on first run.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Pipeline Execution

Run each stage in order from the project root:

```bash
# 1. Variance-filter raw expression data → selects top 1000 genes
python -m src.preprocessing

# 2. Download GPL570 annotation and cross-reference probes
python -m src.annotation

# 3. Compute patient-level Dysregulation Index (DI)
python -m src.dysregulation

# 4. Build pairwise Pearson co-expression edge list (r ≥ 0.70)
python -m src.preprocess_network

# 5. Grid-search all gene pairs for optimal 2D Silhouette separation
#    (computationally intensive — uses multiprocessing)
python -m src.evaluate_pairs

# 6. Launch the interactive dashboard
python -m app.app
```

The dashboard will be available at `http://localhost:8050`.
(Zoom out to 75% to 67% for better view)

## Dashboard Features

| Panel | Description |
|---|---|
| **Joint Gene Phenotyping** | 2D contour/scatter plot for any selected gene pair with Silhouette score |
| **Chromosomal Hotspots** | Genome-wide locus mapping of top-variance probes with interactive click-to-zoom chromosome focus |
| **Co-Expression Network** | Interactive Pearson correlation graph with pathology cohort specific selectors (All vs Subtypes) and click-to-highlight neighbors |
| **Expression Profiles** | Per-gene violin/box plot across all five clinical subtypes with ANOVA |
| **Virtual Expression Assayer** | Softmax centroid classifier simulating expression perturbations in real time |

## Key Extensions & Advanced Features

In addition to the standard pipeline, the platform has been enhanced with:
1. **Interactive Light/Dark Mode Toggle**: A header control that swaps all Plotly layouts and container stylesheets dynamically, with choice persistence stored locally.
2. **Pathology-Cohort Networks**: Switch between the global co-expression network ("All") and subtype-specific edge lists to isolate molecular patterns in individual disease profiles.
3. **Dynamic Chromosome Hotspot Zoom**: Clicking a genomic scatter point on the genome-wide track automatically updates the chromosome selector to focus and center the zoom track on that exact chromosome.
4. **Enhanced Light Theme Aesthetics**: Replaced the default plain gray background with a premium, subtle sky blue (`#EBF3FC`) dashboard background for improved contrast and visual comfort.
   
## Live here
https://oncolens-production.up.railway.app/

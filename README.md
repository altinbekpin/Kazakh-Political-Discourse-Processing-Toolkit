# Kazakh Political Discourse Processing Toolkit

An NLP toolkit for processing Kazakh political discourse: graphematic normalization, morphological analysis, UD-based syntax parsing, synonymization, and classification of discourse tactics such as accusation, criticism, and dispute. Suitable for NLP research and political text analytics.

## Overview

This repository is a work-in-progress implementation of a linguistic processing pipeline for **Kazakh political discourse**. It brings together:

- graphematic and morphological processing for Kazakh,
- syntactic analysis aligned with Universal Dependencies,
- synonymization and lexical normalization in the political domain,
- ontology-driven representation of discourse,
- labeled datasets for discourse tactic classification.

The toolkit is intended for political communication research, computational linguistics, and building intelligent systems for monitoring debates, media, and social networks in Kazakh.

## Repository layout

At the current stage the repository has the following top-level structure:

- `analyzer/` – core resources and code for the linguistic analyzer.
- `analyzer/data/` – data used by the analyzer:
  - **ontology of Kazakh political discourse** (concepts, roles, tactics, relations);
  - **datasets** prepared for training and experimenting in environments such as ChatGPT/Space
    (e.g., labeled examples of accusation, criticism, dispute, etc.).
- `.gitignore` – Git ignore rules.
- `LICENSE` – GPL-3.0 license for the project.

As the project evolves, additional modules for graphematic processing, morphology, syntax, synonymization and classification will be added under `analyzer/`.

## Data: ontology and discourse datasets

The folder [`analyzer/data`](analyzer/data) contains the core resources used in this project:

- **Political discourse ontology**  
  A structured representation of actors (speaker, addressee, target), discourse tactics
  (accusation, criticism, dispute, explanation, manipulation, cooperation, neutral acts),
  events, rhetorical markers, sentiment and modality indicators.  
  This ontology can be used for:
  - enriching model inputs with semantic labels,
  - structuring outputs of language models,
  - supporting downstream political discourse analysis.

- **Discourse tactic datasets**  
  Labeled text samples for:
  - accusation (*айыптау*),
  - criticism (*сынау*),
  - dispute (*дауласу*),
  - and other tactics to be added next.  

  These datasets are constructed from Kazakh news sources using custom crawlers and
  regex-based filters and are intended for training and evaluating classifiers of
  political discourse tactics.

## How to use this repository

At this stage the repository mainly serves as a **data and resource hub**:

- you can explore the ontology and datasets in `analyzer/data/`,
- integrate them into your own NLP or LLM pipelines,
- or use them as training/evaluation material for models that classify discourse strategies.

As code for the analyzer (graphematic processing, morphology, syntax, synonymization, classification) is added and stabilized, usage instructions and examples will be extended in this README.

## License

This project is licensed under the **GPL-3.0** license. See the [`LICENSE`](LICENSE) file for details.

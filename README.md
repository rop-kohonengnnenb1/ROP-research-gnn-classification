# ROP-research-gnn-classification
Research on ROP classification with CNNs and GNNs.

This repository contains research-oriented code for the Retinopathy of Prematurity (ROP) classification, focused on demonstrating model definitions and operational training code and dataset definitions.

The repository shows the KohonenGNNenb1 model based on the CNN and GNN. The GNN is based on the Kohonen SOM. Furthermore the repository shows the Occlusion (perturbation-based) explainability method definition. 

---

## Repository Purpose

The repository is intended **for code overview purposes only**.  

The goals of this repository are:

- To present example **model definitions** for ROP-related classification tasks
- To show the **structure of training and support code** used in research experiments
- To serve as a **reference implementation**, not as a end-to-end pipeline solution for the direct users' usage.

---

## Repository structure

Main script containing the **operational training logic**.  
It serves as the entry point that connects models and utilities.

### `models/`

This directory contains **model definitions**, including architectures used in the research context.

The contents are intended to:
- illustrate how CNN- and GNN-based models are defined
- show structural design choices

### `utils/`

This directory contains **supporting utility code** used by the training script and models.

Typical responsibilities include:
- helper functions
- shared logic required by the training workflow

---

## Usage Scope

This repository is intended for:

- Code inspection
- Structural reference
- Research discussion
- Demonstration of implementation ideas

---

## License

This project is licensed under the **GNU General Public License v3.0**.  
See the `LICENSE` file for details.

---

## Disclaimer

This codebase reflects **research and exploratory work**.  
Interfaces, implementations, and assumptions may change without notice.

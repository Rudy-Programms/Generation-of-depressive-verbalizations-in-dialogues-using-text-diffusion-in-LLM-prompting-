## Generation of depressive verbalizations in dialogues using text-diffusion in LLM prompting 

This repository contains the official implementation of the bachelor thesis: Generation of depressive verbalizations in dialogues using text-diffusion in LLM prompting 

The code used for this project is an adapted version of earlier work by Nöhler et al., which can be found here: *"Text-Diffusion Red-Teaming of Large Language Models: Unveiling Harmful Behaviors with Proximity Constraints"* ([arXiv:2501.08246](https://arxiv.org/pdf/2501.08246)).

This project uses a text-diffusion architecture first introduced in the context of red-teaming to create depressive verbalizations in prompted dialogue, resulting in realistic datasets of psychological markers

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

This project includes several core scripts:

* `Diffusion_Model.py` – Defines the architecture and functionality of the Text Diffusion model.
* `red_team_diffusion.py` – Contains the trainer logic for depressive verbalizations and red-teaming using the diffusion model.
* `rtdConfig.py` – Configuration settings for the experiments.
* `train_diffusion_model.py` – Training script for the diffusion model, with built-in logging.
* `safety_audit.py` – Uses the trained diffusion model to detect and categorize areas of vulnerability in target models.
* `resource.py` – Windows stub for the Unix only "Resource" module that vec2text imports.
* `rewarding.py` – clarifies the reward structure used (psychological interestingness) for model training, and refers to an LLM (GoogleAI) for judgement of interestingness.

---

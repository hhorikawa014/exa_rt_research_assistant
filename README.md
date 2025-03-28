# Real-time Research Assistant with Exa.ai API

## Overview
In this project, I created a real-time AI research assistant to find the user-defined number of research papers about a user-defined topic and generate a Python code of resulted research papers.
- Motivation: Finding cutting-edge ML research topic is getting harder since it's been updated everyday. Utilizing Exa.ai, it should be possible to find one and generate an initial code to grasp ideas.
- First implement the searching part using Exa.ai API. This API is very flexible, fast, and accurate. Link to Exa.ai: https://exa.ai/
- Build the transformer architecture with dropout customizability from scratch for learning purpose, using PyTorch. Train its encoder architecture with the SciTail dataset to calculate the reliablity score of the content.
- For the code generation part, train the full transformer with the entire CodeSearchNet dataset. Unsurprisingly the outputs for sample python code implementation questions was very messy, so decide to rely on the pretrained model SalesForce CodeGen 350M Mono. This model had the descent outputs, however, 1: it takes longer to predict, 2: it relatively generates longer outputs even for simple questions, and 3: it sometimes outputs sentences not codes. Fine-tune this model with randomly selected 100000 samples (seed=42) from the CodeSearchNet dataset. The fine-tuned model performed more accurately and faster, and had more concise output codes. See model_comparison.ipynb for more comparison detail.

Download links for models:
- fine-tuned pretrained model link: https://drive.google.com/file/d/17ozI9XH2thZ1V3t3Z1Rfbyec6oFpDegT/view?usp=drive_link 
- trained transformer model link: https://drive.google.com/file/d/1hH2cRaIefkkmDyAhlBJtS2DAahFDYJRq/view?usp=drive_link
- trained encoder model link: https://drive.google.com/file/d/1Ub799Sae5WnxK7VqYoB46C0DjFzi6XTO/view?usp=drive_link

Note: To reproduce this project, you have to download models above as well as this repository and place them inside the directly "models".

Note: Run this project on VS Code, and run the ipynb files on Google Colab with GPU access (at least better than or equivalent to A100 for trainings. cpu is capable to run just for model comparison).

Note: You need an API key from Exa.ai. Replace "Your EXA API key" in line 10 of exa_search.py with your own API key.

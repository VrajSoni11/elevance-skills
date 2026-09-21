# Elevance Skills — Text-to-Image Generation Internship

A Data Science internship project building a complete **text-to-image generation pipeline** from the ground up — from raw dataset exploration to a fine-tuned Stable Diffusion model — using the **Oxford-102 Flowers** dataset throughout.

Given a text prompt like *"a photo of a red rose"*, this project generates a matching flower image, built entirely from six interconnected tasks that mirror how real text-to-image systems (DALL·E, Stable Diffusion) are structured under the hood.

---

## Table of Contents

- [Overview](#-overview)
- [What This Project Builds](#-what-this-project-builds)
- [Repository Structure](#-repository-structure)
- [The Six Tasks](#-the-six-tasks)
- [Tech Stack](#-tech-stack)
- [Dataset & Captioning Approach](#-dataset--captioning-approach)

---

## Overview

This repository documents a one-month internship project structured around six tasks that, together, form one coherent system rather than six disconnected exercises:

> **Text in → Dataset understanding → Text embeddings → GAN-based generation → Generated flower image out**

Each task was built, debugged, and documented individually, then integrated in **Task 6** into a single end-to-end pipeline. **Task 1** separately applies the same text-to-image concept to a real, production-grade pretrained model (Stable Diffusion) via LoRA fine-tuning.

## What This Project Builds

By the end of this project, the repository contains:

- A fully explored and characterized image dataset with generated captions
- A text preprocessing + embedding pipeline built on CLIP
- A from-scratch **Conditional GAN** (label → image)
- A **self-attention-enhanced GAN** extending the CGAN
- A complete **text-to-image pipeline** accepting free-form prompts
- A **LoRA fine-tuned Stable Diffusion model** with demonstrated before/after effects

---

## The Six Tasks

Tasks were **built in dependency order**, not numeric order, since several depend on outputs from others:

| Order | Task | What It Does | Depends On |
|:---:|---|---|---|
| 1 | **Task 4** — Dataset Exploration | Loads Oxford-102, analyzes class balance/resolution, generates captions | — |
| 2 | **Task 3** — Text Preprocessing | Tokenizes & encodes captions into CLIP embeddings | Task 4 |
| 3 | **Task 2** — Conditional GAN | Generates images conditioned on class labels | Task 4 |
| 4 | **Task 5** — Attention-Enhanced GAN | Adds self-attention to the CGAN architecture | Task 2 |
| 5 | **Task 6** — Full Pipeline | Text → CLIP embedding → GAN image, for *any* prompt | Tasks 3 & 5 |
| 6 | **Task 1** — Diffusion Fine-Tuning | LoRA fine-tunes Stable Diffusion on a custom flower subset | — (independent) |

<details>
<summary><b>Task 4 — Dataset Exploration</b></summary>

- Loads all 8,189 images across 102 flower classes via `torchvision.datasets.Flowers102`
- Maps numeric labels to human-readable class names
- Generates templated natural-language captions (Oxford-102 has no built-in captions)
- Analyzes class distribution, image resolution, and caption length
- Visualizes sample images alongside their captions
- Exports a reusable metadata CSV consumed by later tasks

</details>

<details>
<summary><b>Task 3 — Text Preprocessing with HuggingFace Transformers</b></summary>

- Cleans and tokenizes captions using CLIP's tokenizer (`openai/clip-vit-base-patch32`)
- Analyzes token length distribution to pick a sensible max length
- Encodes all captions into dense embeddings using CLIP's frozen text encoder
- Validates embedding quality via cosine similarity between related/unrelated flower captions
- Exports embeddings for direct reuse in Task 6

</details>

<details>
<summary><b>Task 2 — Conditional GAN (CGAN)</b></summary>

- DCGAN-style Generator + Discriminator, conditioned on class label via embeddings
- Trained adversarially on 64×64 flower images
- Tracks fixed noise + fixed classes across training for visual progress comparison
- Demonstrates that different class labels produce visibly different generated images

</details>

<details>
<summary><b>Task 5 — Attention-Enhanced GAN</b></summary>

- Extends Task 2's architecture with a **SAGAN-style self-attention module**
- Attention inserted at the 16×16 feature-map resolution in both networks
- Everything else kept identical to Task 2 for a controlled comparison
- Includes attention map visualization — directly inspecting what the model attends to

</details>

<details>
<summary><b>Task 6 — Full Text-to-Image Pipeline</b></summary>

- Combines Task 3's frozen CLIP encoder with an adapted version of Task 5's attention GAN
- **Key change:** conditions on continuous CLIP embeddings instead of discrete class labels — enabling arbitrary free-form text prompts, not just the 102 fixed classes
- Trains end-to-end with live-encoded captions per batch
- Ships a single `text_to_image(prompt)` function as the final deliverable

</details>

<details>
<summary><b>Task 1 — Fine-Tuning Stable Diffusion (LoRA)</b></summary>

- Fine-tunes `runwayml/stable-diffusion-v1-5` using **LoRA** (Low-Rank Adaptation)
- Only 0.093% of parameters (797K / 860M) are trainable — the rest of the model stays frozen
- Custom dataset: 45 images across 3 flower classes (sunflower, rose, water lily)
- Generates before/after comparisons with identical prompts and seeds to isolate the fine-tuning effect
- Final LoRA weights: **3.1 MB**, vs. several GB for the full base model

</details>

---

## Tech Stack

| Category | Tools |
|---|---|
| Compute | Google Colab (free tier, NVIDIA T4 GPU) |
| Local dev | VS Code |
| Core ML | PyTorch |
| NLP / Embeddings | HuggingFace Transformers (CLIP) |
| Diffusion fine-tuning | HuggingFace Diffusers + PEFT (LoRA) |
| Dataset | Oxford-102 Flowers (`torchvision.datasets.Flowers102`) |
| Visualization | Matplotlib, Seaborn |
| Storage | Google Drive (persistent artifacts across Colab sessions) |

---

## Dataset & Captioning Approach

**Oxford-102 Flowers** is used consistently across Tasks 2, 4, 5, and 6. Since this dataset has no built-in natural-language captions (unlike COCO), a **templated caption generator** produces varied descriptions from each image's class name:

```
"a photo of a {flower}"
"a close-up photo of a {flower} flower"
"a beautiful {flower} in bloom"
"an image showing a {flower} flower"
"a {flower} flower with vivid petals"
```

This is a standard, transparent approach for adapting label-only datasets to text-conditioned generation work — documented here explicitly rather than disguised as real captions.

Task 1 uses a small curated subset (45 images, 3 classes) for LoRA fine-tuning.

---

## Results Summary

| Task | Key Result |
|---|---|
| Task 4 | 8,189 images across 102 classes characterized; class imbalance and resolution variance documented |
| Task 3 | All captions encoded into 512-dim CLIP embeddings; similarity checks confirm meaningful representations |
| Task 2 | Clear visual improvement from early → late training; distinct outputs per class label |
| Task 5 | Self-attention successfully integrated; attention maps show interpretable spatial focus |
| Task 6 | Working `text_to_image()` pipeline responds to free-form prompts beyond the 102 fixed classes |
| Task 1 | Visible before/after shift in generated images after LoRA fine-tuning; 3.1 MB adapter vs. multi-GB base model |

---

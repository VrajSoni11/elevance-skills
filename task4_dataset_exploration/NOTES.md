# Task 4: Dataset Exploration (Oxford-102 Flowers)

## Objective
Load and examine the Oxford-102 Flowers dataset to understand its structure —
number of classes, class balance, image resolution, and text description
characteristics — as a foundation for the later text-to-image tasks (2, 3, 5, 6).

## Approach
- Loaded the dataset using `torchvision.datasets.Flowers102` (train/val/test
  splits), which downloads directly from the official Oxford VGG source.
- Mapped numeric labels (0–101) to human-readable flower names using the
  official Oxford-102 category list.
- Since Oxford-102 has no built-in natural-language captions, generated
  templated text captions per image (e.g. "a photo of a pink primrose",
  "a close-up photo of a sweet pea flower") using 5 varying templates. This is
  a standard, transparent workaround for label-only datasets and is reused in
  Tasks 3, 5, and 6.
- Computed and visualized:
  - Number of images per class (class balance/imbalance)
  - Image resolution distribution (sampled 300 images)
  - Caption length distribution (word count)
  - Sample images displayed alongside their generated captions
- Exported a reusable `oxford102_metadata.csv` (split, label, class name,
  caption) so later tasks don't need to redo this setup.

## Results
- Dataset loaded successfully across all three splits; total ~8,189 images
  across 102 flower classes.
- Class counts are imbalanced — some classes have far fewer images than
  others (see `class_distribution.png` for the exact per-class breakdown).
- Image resolutions vary significantly since the dataset isn't pre-resized;
  most images fall in a moderate range but there's real variance (see
  `resolution_distribution.png`).
- Templated captions average a small, consistent word count, giving a clean
  and uniform text input for tokenization work in Task 3.
- Sample image+caption grid confirms captions are sensibly matched to their
  images.

## Challenges & how they were resolved
- Oxford-102 has no ready-made captions unlike COCO — resolved by building a
  templated caption generator instead of treating this as a blocker.
- Checking resolution across all ~8,189 images would be slow — resolved by
  statistically sampling 300 images, which is sufficient to characterize the
  distribution.

## Next steps
- Use `oxford102_metadata.csv` and the caption generator in Task 3 to build
  the tokenization/encoding pipeline.
- Reuse the same dataset and class structure in Task 2 (CGAN) for
  label-conditioned image generation.
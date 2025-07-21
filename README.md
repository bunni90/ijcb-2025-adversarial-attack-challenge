# 2025 Adversarial Attack Challenge
Team-Roma submission for the 2025 Adversarial Attack Challenge for Secure Face Recognition (AAC).
https://www.youverse.id/adversarial

## Resilience Track
Our solution relied on FaceNet trained on CASIA-WebFace combined with Gaussian blurring as a preprocessing defense. By testing various image transformations (flipping, JPEG compression, median blur, etc.), we found that a simple (3×3) Gaussian blur significantly boosted robustness under FGSM-style attacks

## Detection Track
We developed a custom ResNet18MoreThanRGB architecture, enriched with features like JPEG recompression, grayscaling, morphological ops, and scaling artifacts. Trained from scratch on a class-weighted loss.

## Contributors:
Niklas Bunzel,
Lukas Graner,
Nicholas Göller

# NYCU Computer Vision 2026 HW3: Instance Segmentation

**Student ID:** 314561003  
**Name:** Basetti Sai Viswas  

## Introduction
This repository contains my implementation for the NYCU 2026 Computer Vision Homework 3: Instance Segmentation. The goal of this project is to accurately segment four distinct types of cells in phase-contrast microscopy images. 

I utilized a pure-vision **Mask R-CNN** architecture with a `ResNet50-FPN` backbone. To overcome the high frequency of false positives and severe class imbalances (specifically regarding Class 3 cells), I implemented custom aspect-ratio scaling to prevent OOM errors, a strict probability thresholding mechanism, and an IoU-based confusion matrix evaluation to optimize the COCO mAP score.

## Environment Setup
This project was developed and tested using Python 3.9+. It is recommended to run this code within a virtual environment (e.g., Conda or Venv).

1. Clone the repository:
   ```bash
   git clone https://github.com/saiviswasb/CV-HW3-Instance-Segmentation.git
   cd CV-HW3-Instance-Segmentation
2. Install the required dependencies:

   ```bash
   pip install torch torchvision numpy opencv-python pycocotools matplotlib seaborn

3. Usage
Ensure that the dataset is extracted and placed in a folder named hw3-data-release in the same directory as the script.

Training, Inference, and Evaluation
The main.py script is modularized to handle the entire pipeline end-to-end. Running the script will automatically train the model for 20 epochs, generate the test-results.json required for CodaBench, and run the additional experiments.
     
   
    python main.py
4. Outputs:
Upon completion, the script will create an ./outputs/ directory containing:

a.)maskrcnn_baseline_20ep.pth (The trained model weights)

b.)DL_HW3_Submission.zip (The RLE formatted submission file)

c.)Exp1_Threshold.png (Threshold ablation graph)

d.)Exp2_Matrix.png (IoU Confusion Matrix)

e.)Exp3_Visual.png (Ground truth vs. Prediction visualization)

## Results & Visualizations
Here are the visualizations from the additional experiments detailing the impact of confidence thresholding and class-weighted sampling.

### <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/05f5fde6-27ff-4350-a554-96e4c29f7f3b" />


### <img width="751" height="590" alt="image" src="https://github.com/user-attachments/assets/218dc825-9fac-41b1-b5f2-5bac7414385a" />


### <img width="1382" height="712" alt="image" src="https://github.com/user-attachments/assets/65606816-0fbf-420a-819c-c4084c3cec01" />

## Performance Snapshot

Below is the CodaBench leaderboard snapshot reflecting the optimized Mask R-CNN baseline.
<img width="1442" height="60" alt="image" src="https://github.com/user-attachments/assets/157e437c-e463-4b6d-9067-a7c75258cc5e" />

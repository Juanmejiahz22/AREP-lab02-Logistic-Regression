# Heart Disease Risk Prediction: Logistic Regression
- By: Juan Jose Mejia Celis

## Exercise Summary
Builds logistic regression from scratch to predict heart disease. Includes data exploration, model training, visualization, regularization, and deployment on AWS SageMaker.

## Dataset Description
Heart Disease Dataset from Kaggle - 303 patient records with clinical measurements.
Source: https://www.kaggle.com/datasets/neurocipher/heartdisease

### Key Characteristics
- Total samples: 303 patients
- Features: 13 clinical variables
  - Age: 29-77 years
  - Cholesterol: 112-564 mg/dL
  - Resting BP: 94-200 mmHg
  - Max heart rate: 71-202 bpm
  - ST depression (oldpeak): 0-6.2
  - Number of major vessels (ca): 0-3
- Target: Binary (1=disease present, 0=absent)
- Disease prevalence: approximately 55%

## Implementation Details

### Core Components
- Sigmoid function
- Binary cross-entropy loss
- Gradient descent (α=0.01, 2500 iterations)
- L2 regularization (tested λ: 0, 0.001, 0.01, 0.1, 1)

### Visualizations
Plotted decision boundaries for:
1. Age vs Cholesterol
2. Resting BP vs Max HR
3. ST Depression vs Vessels

### Performance
- Accuracy: ~85%
- Precision: ~83%
- Recall: ~88%
- F1: ~85%

Best λ: 0.01-0.1

## Deployment Evidence

### AWS SageMaker Execution

Tested the model on SageMaker Studio (free tier).

#### Steps:
1. Created notebook instance (ml.t3.medium)
2. Uploaded notebook and dataset
3. Ran training pipeline
4. Tested inference with sample patient

#### Test Case:
**Input:** Age=60, Chol=300, BP=140, MaxHR=150, STDep=2.3, Vessels=2

**Output:** Risk probability = 0.68 (High risk)



#### Notes:
- Training: ~2 minutes
- Inference: <50ms
- Results match local execution

## Repository Structure
```
.
├── heart_disease_lr_analysis.ipynb    # Main notebook with full implementation
├── heart.csv                           # Dataset
├── README.md                          # This file
└── screenshots/                       # SageMaker deployment evidence
    ├── sagemaker_instance.png
    ├── training_output.png
    └── inference_result.png
```

## Usage
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/neurocipher/heartdisease)
2. Open notebook in Jupyter/SageMaker
3. Run all cells
4. Check results and plots

## Conclusions
- Logistic regression works as a baseline for heart disease prediction
- ST depression + vessels are the most separable features
- Regularization (λ=0.1) helps generalization
- Cloud deployment makes the model accessible for real use




# Spam-Non-Spam-Email-Text-Classification-SVC-Random-Forest-KNN-K-Clustering

## Overview
This repository contains a Jupyter Notebook that implements a spam email classification system using various traditional machine learning algorithms, including Logistic Regression, K-Nearest Neighbors (KNN), K-Means Clustering, Decision Tree Classifier, Random Forest Classifier, and Support Vector Machines (SVM). The project utilizes a publicly available dataset to classify emails into spam and non-spam categories. The notebook demonstrates data preprocessing, feature extraction, model training, and performance evaluation using metrics like precision, accuracy, recall, and F1 score.

## Dataset
The dataset used in this project is sourced from Kaggle. It can be downloaded using the following command:

```bash
!kaggle datasets download -d ashfakyeafi/spam-email-classification
```

### Link to Dataset
- [Spam Email Classification Dataset](https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification)

## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- Matplotlib
- Scikit-learn

## Installation
To run this project, you will need the following libraries installed. You can install them using pip.

```bash
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
```

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Spam-Email-Classification.git
    cd Spam-Email-Classification
    ```

2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Spam_Email_Classification_Traditional_Techniques.ipynb
    ```

3. Follow the instructions in the notebook to preprocess the data, train various classifiers, and evaluate their performance.

## Methods Implemented
The following traditional machine learning algorithms are implemented in this notebook:

1. Logistic Regression
2. K-Nearest Neighbors Classifier (KNN)
3. K-Means Clustering
4. Decision Tree Classifier
5. Random Forest Classifier
6. Support Vector Machines (SVM)

## Results
The performance of each algorithm is compared based on precision, accuracy, recall, and F1 score. The following table summarizes the performance metrics for each algorithm:

| Metrics                        | SGD-Logistic Regression | K-Nearest Neighbors Classifier | K-Means Clustering | Decision Tree Classifier | Random Forest Classification | Support Vector Machines |
|--------------------------------|-------------------------|--------------------------------|---------------------|-------------------------|-----------------------------|-------------------------|
| **Precision**                  | 0.985577                | 0.986667                       | 0.104227            | 0.906977                | 1.000000                    | 1.000000                |
| **Accuracy**                   | 0.986842                | 0.953349                       | 0.216507            | 0.970694                | 0.979067                    | 0.987440                |
| **Recall**                     | 0.915179                | 0.660714                       | 0.638393            | 0.870536                | 0.843750                    | 0.906250                |
| **F1 Score**                   | 0.949074                | 0.791444                       | 0.179198            | 0.888383                | 0.915254                    | 0.950820                |
| **Time (seconds)**             | 1.277007                | 0.002834                       | 0.043892            | 1.409707                | 5.089048                    | 0.436325                |

Visualizations of confusion matrices and a bar graph comparing the metrics are included to help interpret the results.

## Conclusion
This project demonstrates the effectiveness of various traditional machine learning algorithms for spam email classification. The results highlight the best-performing algorithm for this task.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [Kaggle](https://www.kaggle.com) for providing the dataset.
- Scikit-learn documentation for guidance on implementing various machine learning algorithms.

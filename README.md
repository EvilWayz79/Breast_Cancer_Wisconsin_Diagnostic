# Breast_Cancer_Wisconsin_Diagnostic

## Rubén Recalde


This repository contains an analysis of the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to predict whether breast masses are benign or malignant based on various features computed from digitized images of fine needle aspirate (FNA) samples.


### Methods Used:

* Multi Layer Perceptron    MLP
* Random Forest             RF
* Logistic Regression         LR

we will follow the next algorithm for the data analysis:

* Import modules
* Import dataset from sklearn.datasets -- if you want to use the online version of the dataset
* Load dataset
* Data process
    * delete NaN column
    * Transform target column to binary
    * Separate data from target
* Check results
* Split the data into train and test subsets
* Normalize the features in order for the model to have better performance and avoid bias by features that have larger values.
* Set up data in values between 0 and 1


### Define the MLP (Multi Layer Perceptron) model

* The choice of the number of layers, neurons, activation functions, and hyperparameters can affect the performance and generalization of the model. I decided on these values based  on some common practices and heuristics for MLP design, such as:
    * Using a sigmoid activation function for the output layer, since this is a binary classification problem.
    * Using a sigmoid activation function for the hidden layers, since this is a simple and smooth nonlinear function that can approximate any function.
    * Using a binary cross-entropy loss function, since this is a suitable loss for binary classification problems.
    * Using an adam optimizer, since this is a popular and efficient gradient-based optimization algorithm that can adapt the learning rate dynamically.
    * Using a small number of hidden layers (three) and neurons (16, 8, and 4), since this is a relatively small and low-dimensional dataset (569 samples and 30 features), and a complex model might overfit the data.
    * Using a validation split of 0.2, since this is a reasonable proportion of the data to use for evaluating the model during training.
    * Using 100 epochs and 32 batch size, since these are typical values that can allow the model to converge without taking too long.

* Compile the model
* Train the model
* Evaluate the model
* Generate predictions
* Generate the confusion matrix
* Print the confusion matrix
* Plot the confusion matrix
* Get the Precision, which is the proportion of positive predictions that are actually positive
* Get the F1 score, which is the harmonic mean or precision and recall that measures the balance between them
* Get the recall score, which os the proportion of positive samples that are correctly predicted


### Conclusions from MLP analysis

Based on these results for the confusion matrix, I would conclude that the classification model is very accurate and has high performance on the test set.

### Results analysis

* True Positives (TP): The model accurately   predicted 70 malignant cases (actual positives).
* True Negatives (TN): The model accurately   predicted 41 benign cases (actual negatives).
* False Positives (FP): The model incorrectly predicted 2 benign cases as malignant.
* False Negatives (FN): The model incorrectly predicted 1 malignant case as benign.

* Accuracy 0.973 this means that the model predicted correctly 97.37% of the test samples
* Precision 
    * 0.9722 for positive (malignant) calculated by the formula TP/(TP + FP) giving a 97.22% of correctly predicted malignant samples
    * 0.9762 for negative (benignant) calculated by the formula TN/(TN + FN) giving a 97.62% of correctly predicted benign samples
* Recall 
    * 0.9859 Proportion of positive samples (malignant) that are correctly determined by the formula TP/(TP + FN) meaning 98.58% correct predictions
    * 0.9535 Proportion of negative samples (benign)    that are correctly determined by the formula FN/(FN + TN) meaning 95.35% correct predictions
* F1 score: The harmonic mean of precision and recall that measures the balance between them calculated by: 2 * (precision * recall) / (precision + recall)
    * positive (malignant): 0.9790 which means there's a high balance between precision and recall for the malignan samples
    * negative (benign):    0.9648 which means there's a high balance between precision and recall for the benign   samples

* The MLP approach to this dataset yields a very good overall score. The model effectively balances precision and recall, making it a robust choice for breast cancer prediction.

### Random Forest Approach RF
* We already have the dataset, X (data) and y (target)  test and training are already separated too. 
* ready to work!
* Initialize the Random Forest Classifier
* Train the model
* Make predictions on the test set
* Evaluate the model
* Generate the confusion matrix
* Display the confusion matrix
* Calculate precision score
* Calculate Recall and f1
* Get the confusion matrix
* Print the confusion matrix

### Conclusions from Random Forest analysis

Based on these results for the confusion matrix, I would conclude that the classification model is very accurate and has high performance on the test set.

### Results analysis

* True Positives (TP): The model accurately  predicted 70 malignant cases (actual positives).
* True Negatives (TN): The model accurately  predicted 40 benign cases (actual negatives).
* False Positives (FP): The model incorrectly predicted 3 benign cases as malignant.
* False Negatives (FN): The model incorrectly predicted 1 malignant case as benign.

* Accuracy 0.96 this means that the model predicted correctly 96% of the test samples
* Precision 
    * 0.95 for positive (malignant) calculated by the formula TP/(TP + FP) giving a 95% of correctly predicted malignant samples
    * 0.9756 for negative (benignant) calculated by the formula TN/(TN + FN) giving a 97.56% of correctly predicted benign samples
* Recall 
    * 0.9859 Proportion of positive samples (malignant) that are correctly determined by the formula TP/(TP + FN) meaning 98.59% correct predictions
    * 0.9756 Proportion of negative samples (benign)    that are correctly determined by the formula TN/(TN + FN) meaning 97.56% correct predictions
* F1 score: The harmonic mean of precision and recall that measures the balance between them calculated by: 2 * (precision * recall) / (precision + recall)
    * positive (malignant): 0.9722 which means there's a 97.22% balance between precision and recall for the malignan samples
    * negative (benign):    0.9756 which means there's a 97.56% balance between precision and recall for the benign   samples

* The Random Forest approach to this dataset yields a very good overall score. The model effectively balances precision and recall, making it a robust choice for breast cancer prediction.


### Logistic Regression LR

* Using previously stated train and test data
* Generate model
* Train LR model
* Predictions
* Calculate accuracy
* Calculate confusion matrix
* Display confusion matrix
* Calculate precision
* Calculate recall and f1
* Calculate confusion matrix for LR
* Plot confusion matrix for LR

### Conclusions from Logistic Regression analysis

Based on these results for the confusion matrix, I would conclude that the classification model is very accurate and has high performance on the test set.

### Results analysis

* True Positives (TP): The model correctly predicted 70 malignant cases (actual positives).
* True Negatives (TN): The model correctly predicted 41 benign cases (actual negatives).
* False Positives (FP): The model incorrectly predicted 2 benign cases as malignant.
* False Negatives (FN): The model incorrectly predicted 0 malignant case as benign.

* Accuracy 0.98 this means that the model predicted correctly 98% of the test samples
* Precision 
    * 0.97 for positive (malignant) calculated by the formula TP/(TP + FP) giving a 97% of correctly predicted malignant samples
    * 1 for negative (benignant) calculated by the formula TN/(TN + FN) giving a 100% of correctly predicted benign samples
* Recall 
    * 1 Proportion of positive samples (malignant) that are correctly determined by the formula TP/(TP + FN) meaning 100% correct predictions
    * 1 Proportion of negative samples (benign)    that are correctly determined by the formula TN/(TN + FN) meaning 100% correct predictions
* F1 score: The harmonic mean of precision and recall that measures the balance between them calculated by: 2 * (precision * recall) / (precision + recall)
    * positive (malignant): 0.9861 which means there's a 98.61% balance between precision and recall for the malignan samples
    * negative (benign):    1 which means there's a 100% balance between precision and recall for the benign   samples

* The Linear approach to this dataset yields a perfect score for screening cancer tests as it would not confuse FN diagnosis. The abscense of false negatives is crucial for patient survival

### Results Summary

**Confusion Matrix Summary**
| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| MLP   | 41 |	2 |	 1 | 70 |
| RF    | 40 |  3 |  1 | 70 |
| LR    | 41 |  2 |  0 | 71 |


**Model Index Summary**

| Model | Accuracy | Precision+ | Recall+ |    f1+   |
|-------|----------|------------|---------|----------|
| MLP   | 0.9737   |	0.9722  |  0.9859 |  0.9790  |
| RF    | 0.9649   |	0.9589  |  0.9859 |  0.9722  |
| LR    | 0.9825   |	0.9726  |  1.0000 |  0.9861  |


### FINAL CONCLUSIONS

Based on the gathered results we can assert the following:

* All three models perform well, but let’s focus on minimizing false negatives (FN) since missing malignant cases is critical.
* Logistic Regression (LR) stands out:
    * It achieves perfect recall (100%) for malignant cases, meaning it correctly identifies all malignant samples.
    * The F1 score (harmonic mean of precision and recall) is also high (0.9861).
    * LR has the least combination between FP and FN (2, 0) from the models
* Therefore, Logistic Regression appears to be the best model for the proposed Breast Cancer Wisconsin dataset, emphasizing its ability to minimize false negatives and accurately predict malignant cases.

### Recommendations

1. **Model Selection and Validation:**
Continue using the Logistic Regression (LR) model as it performed well in terms of recall, F1 score, and minimizing false negatives.
Validate the LR model on additional independent populations to ensure its robustness across different patient groups.
2. **Feature Engineering:**
Explore additional features beyond clinico-pathological factors. Consider incorporating genetic information (e.g., single-nucleotide polymorphisms) to enhance model performance.
3. **Precision Medicine:**
Investigate precision screening approaches that consider individual risk factors and genetic profiles.
Evaluate the impact of precision treatment paradigms (adjuvant, neo-adjuvant, and metastatic) on breast cancer outcomes.
4. **Population-Specific Models:**
Develop models tailored to specific patient populations (e.g., high-risk patients, young or elderly patients).
Consider age, ethnicity, and other demographic factors when designing and validating models.
5. **Long-Term Monitoring:**
Implement long-term monitoring of model performance in real-world clinical settings.
Regularly update and revalidate models as new data becomes available.
6. **Collaboration and Data Sharing:**
Collaborate with other researchers and institutions to pool data and improve model generalization.
Share model results and insights to contribute to breast cancer research.
Remember that breast cancer is a complex disease, and continuous research and collaboration are essential for improving prediction models and patient outcomes.
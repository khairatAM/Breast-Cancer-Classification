# Breast-Cancer-Classification
Invasive ductal carcinoma (IDC) is the most common type of breast cancer. This project employs deep learning to classify histology images as being either positive or negative samples.

**Dataset**: 3GB of 50x50 breast cancer samples scanned at 40x with 198,738 IDC negative and 78,786 IDC positive. (Link: https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images).

**Key learning points:**
1. Preprocessing image datasets from local directory via os, imutils and Keras packages in Python.
2. Image augmentation in Keras during training to improve model robustness.
3. Balancing the training loss of an unbalanced dataset using class weighting in Keras.
4. Familiar with CNNs but learned about Seperable Convolutions for faster and more efficient computation.
5. Improved proficiency with writing modules, classes, methods and file I/O in Python.
6. Handled overfitting of model by increasing kernel size and number of filters, and decreasing the learning rate.

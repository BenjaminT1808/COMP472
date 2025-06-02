import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the greyscale digits dataset
# This dataset contains images of handwritten digits (0-9)
digits = datasets.load_digits()

# Display the first 10 images from the dataset for training
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(18, 3))
fig.suptitle('First Ten Dataset Images')
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

data=digits.data

# Create a Support Vector Machine classifier
clf = svm.SVC(gamma=0.001)

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels for the test set
SVMPredicted = clf.predict(X_test)

# Evaluate score 
score = clf.score(X_test, y_test)
print("SVM score: ", score)

# Display the first 10 images from the test set with their SVM predicted labels
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(18, 3))
fig.suptitle('First Ten Dataset Images SVM Predicted')
for ax, image, prediction in zip(axes, X_test, SVMPredicted):
    ax.set_axis_off()
    ax.imshow(image.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Predicted: {prediction}")

# Display the classification report as a figure
report = metrics.classification_report(y_test, SVMPredicted)
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('SVM Classification Report')
ax.axis('off')
ax.text(0, 1, f"SVM Classification Report\n\n{report}", fontsize=12, va='top', family='monospace')
plt.show(block=False)

# Display the confusion matrix as a figure
disp = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, SVMPredicted, display_labels=digits.target_names, cmap=plt.cm.Blues
)
disp.figure_.suptitle("SVM Confusion Matrix")
print(f"SVM Confusion matrix:\n{disp.confusion_matrix}\n")
plt.show(block=False)




# Create a Logistic Regression model for predicting the data 
model = LogisticRegression(solver='lbfgs')

# Training 
model.fit(X_train, y_train)

# Prediction
LogisticPredicted = model.predict(X_test)

# Evaluate score
score = model.score(X_test, y_test)
print("LR Score: ", score)

# Display the first 10 images from the test set with their Logistic Regression predicted labels
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(18, 3))
fig.suptitle('First Ten Dataset Images Logistic Regression Predicted')
for ax, image, prediction in zip(axes, X_test, LogisticPredicted):
    ax.set_axis_off()
    ax.imshow(image.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Predicted: {prediction}")

# Display the classification report as a figure
report = metrics.classification_report(y_test, LogisticPredicted)
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Logistic Regression Classification Report')
ax.axis('off')
ax.text(0, 1, f"Logistic Regression Classification Report\n\n{report}", fontsize=12, va='top', family='monospace')
plt.show(block=False)

# Display the confusion matrix as a figure
disp = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, LogisticPredicted, display_labels=digits.target_names, cmap=plt.cm.Blues
)
disp.figure_.suptitle("Logistic Regression Confusion Matrix")
print(f"LR Confusion matrix:\n{disp.confusion_matrix}\n")
plt.show()

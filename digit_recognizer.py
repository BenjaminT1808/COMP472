import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

# Load the greyscale digits dataset
# This dataset contains images of handwritten digits (0-9)
digits = datasets.load_digits()

# Display the first 10 images from the dataset for training
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(18, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

# flattens the images to a 2D array
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a Support Vector Machine classifier
clf = svm.SVC(gamma=0.001)

# Split the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the labels for the test set
predicted = clf.predict(X_test)

# Display the first 10 images from the test set with their predicted labels
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(18, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    ax.imshow(image.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Predicted: {prediction}")

# Display the classification report as a figure
report = metrics.classification_report(y_test, predicted)
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
ax.text(0, 1, f"Classification Report\n\n{report}", fontsize=12, va='top', family='monospace')
plt.show(block=False)

# Display the confusion matrix as a figure
disp = metrics.ConfusionMatrixDisplay.from_predictions(
    y_test, predicted, display_labels=digits.target_names, cmap=plt.cm.Blues
)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}\n")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize  # Corrected import
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('mental_health_dataset.csv')

# Define the recommendation function based on mental health rules
def mental_health_recommendation(row):
    condition = row['Mental_Health_Condition']
    severity = row['Severity']
    stress_level = row['Stress_Level']
    sleep_hours = row['Sleep_Hours']
   
    # Rule 1: High Severity
    if condition == 'Yes' and severity == 'High':
        return "Seek immediate professional help."
   
    # Rule 2: Medium Severity
    elif condition == 'Yes' and severity == 'Medium':
        if stress_level in ['High', 'Medium']:
            return "Consider professional consultation."
        elif stress_level == 'Low':
            return "Monitor condition and consider follow-up consultation."
   
    # Rule 3: Low Severity
    elif condition == 'Yes' and severity == 'Low':
        if stress_level == 'High' and sleep_hours < 6:
            return "Seek professional help due to stress and poor sleep."
        elif stress_level == 'Medium' and 6 <= sleep_hours <= 8:
            return "Try self-care strategies like exercise and relaxation."
        elif stress_level == 'Low' and sleep_hours > 8:
            return "Maintain self-care and regular check-ups."
   
    # Rule 4: No Mental Health Condition
    elif condition == 'No':
        if stress_level in ['High', 'Medium']:
            return "Monitor stress and adopt stress-relieving techniques."
        if stress_level == 'Low':
            return "Maintain wellness practices to preserve mental health."
        if sleep_hours < 6 and stress_level == 'High':
            return "Consider seeking professional consultation for stress and sleep issues."

    return "No action needed."

# Apply the recommendation function to each row in the DataFrame
df['Recommendation'] = df.apply(mental_health_recommendation, axis=1)

# Encoding 'Mental_Health_Condition', 'Severity', and 'Stress_Level' into numeric values
label_encoder = LabelEncoder()

df['Mental_Health_Condition'] = label_encoder.fit_transform(df['Mental_Health_Condition'])
df['Severity'] = label_encoder.fit_transform(df['Severity'])
df['Stress_Level'] = label_encoder.fit_transform(df['Stress_Level'])
df['Recommendation'] = label_encoder.fit_transform(df['Recommendation'])  # Now we encode the Recommendation column

# Split dataset into features and target
X = df[['Mental_Health_Condition', 'Severity', 'Stress_Level', 'Sleep_Hours']]  # Features
y = df['Recommendation']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict recommendations on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Calculate precision for each class
precision = precision_score(y_test, y_pred, average=None, labels=range(len(label_encoder.classes_)), zero_division=0)

# Print precision for each class
print("\nPrecision for each class:")
for rec, prec in zip(label_encoder.classes_, precision):
    print(f"{rec}: {prec:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))

# Plot confusion matrix using Seaborn heatmap for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Recommendation')
plt.ylabel('Expected Recommendation')
plt.title('Confusion Matrix')
plt.show()

# Binarize the Expected and Predicted recommendations for ROC curve
y_true = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
y_pred_binarized = label_binarize(y_pred, classes=range(len(label_encoder.classes_)))

# Plot ROC curve for each class (one-vs-rest)
plt.figure(figsize=(10, 8))
for i, recommendation in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_binarized[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve for {recommendation} (AUC = {roc_auc:.2f})')

# Plot settings for ROC curve
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
plt.legend(loc='lower right')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('mental_health_dataset.csv')

# Function for manually defining the ground truth (True_Mental_State)
def determine_mental_state(stress_level, sleep_hours):
    stress_level = stress_level.strip().lower()
   
    if stress_level == 'high' and sleep_hours < 6:
        return 'Not Mentally Healthy'
    elif stress_level == 'moderate' and 6 <= sleep_hours < 8:
        return 'Mentally Healthy'
    elif stress_level == 'low' or sleep_hours >= 8:
        return 'Mentally Healthy'
    else:
        return 'Not Mentally Healthy'

# Manually define the True_Mental_State column based on the rules
df['Mental_State'] = df.apply(lambda row: determine_mental_state(row['Stress_Level'], row['Sleep_Hours']), axis=1)

# Encode categorical variables to numeric
label_encoder = LabelEncoder()

df['Stress_Level'] = label_encoder.fit_transform(df['Stress_Level'].str.strip().str.lower())
df['Mental_State'] = label_encoder.fit_transform(df['Mental_State'])

# Features and target variables
X = df[['Stress_Level', 'Sleep_Hours']]
y = df['Mental_State']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Predict mental state on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate precision
precision = precision_score(y_test, y_pred, pos_label=label_encoder.transform(['Mentally Healthy'])[0])
print(f"Precision: {precision * 100:.2f}%")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test, y_pred
)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Display and plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

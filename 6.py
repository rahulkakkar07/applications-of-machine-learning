# ============================================================
# SPAM EMAIL DETECTION USING MACHINE LEARNING
# Algorithm: Multinomial Naive Bayes
# Library: Scikit-learn
# ============================================================


# ------------------------------------------------------------
# STEP 1: Import Required Libraries
# ------------------------------------------------------------

import pandas as pd

# Used to divide data into training and testing sets
from sklearn.model_selection import train_test_split

# Converts text into numerical form
from sklearn.feature_extraction.text import CountVectorizer

# Naive Bayes classifier for text classification
from sklearn.naive_bayes import MultinomialNB

# Evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ------------------------------------------------------------
# STEP 2: Create a Small Sample Dataset
# ------------------------------------------------------------

# Here:
# ham  = normal / genuine message
# spam = unwanted / promotional message

data = {
    'message': [
        'Congratulations! You won a free prize',
        'Please attend the meeting tomorrow',
        'Claim your free cash reward now',
        'Can we meet for lunch today?',
        'You have won a lottery of 1 lakh rupees',
        'Please submit your assignment by Monday',
        'Get a free mobile phone now',
        'Your class starts at 10 AM tomorrow',
        'Exclusive offer! Buy now and win prizes',
        'Can you send me the project report?',
        'Urgent! You have won a cash reward',
        'Your appointment is confirmed for tomorrow',
        'Click here to claim your free gift',
        'Please call me when you reach home',
        'Win money instantly by clicking this link',
        'The meeting has been rescheduled to 2 PM',
        'You are selected for a free vacation',
        'Please bring your notebook to class',
        'Earn money quickly from home',
        'Happy birthday! Have a wonderful day'
    ],

    'label': [
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham',
        'spam',
        'ham'
    ]
}


# ------------------------------------------------------------
# STEP 3: Convert Dictionary into DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# ------------------------------------------------------------
# STEP 4: Check Dataset Information
# ------------------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nSpam and Ham Message Count:")
print(df['label'].value_counts())


# ------------------------------------------------------------
# STEP 5: Define Input and Output
# ------------------------------------------------------------

# X contains email / message text
X = df['message']

# y contains the corresponding class
# spam or ham
y = df['label']


# ------------------------------------------------------------
# STEP 6: Divide Data into Training and Testing Sets
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,       # 30% testing data
    random_state=42,      # same result every time
    stratify=y            # maintains spam/ham proportion
)

print("\nTraining Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))


# ------------------------------------------------------------
# STEP 7: Convert Text into Numerical Features
# ------------------------------------------------------------

# Machine learning algorithms cannot directly understand text.
# CountVectorizer converts text into numbers by counting words.

vectorizer = CountVectorizer()


# Learn the vocabulary from training data
# and convert training messages into numerical vectors
X_train_vectorized = vectorizer.fit_transform(X_train)


# Convert testing messages using the SAME vocabulary
X_test_vectorized = vectorizer.transform(X_test)


print("\nVocabulary:")
print(vectorizer.get_feature_names_out())


# ------------------------------------------------------------
# STEP 8: Create the Machine Learning Model
# ------------------------------------------------------------

# Multinomial Naive Bayes works very well
# for text classification problems.

model = MultinomialNB()


# ------------------------------------------------------------
# STEP 9: Train the Model
# ------------------------------------------------------------

model.fit(X_train_vectorized, y_train)

print("\nModel Training Completed Successfully.")


# ------------------------------------------------------------
# STEP 10: Predict Test Data
# ------------------------------------------------------------

y_pred = model.predict(X_test_vectorized)

print("\nActual Labels:")
print(list(y_test))

print("\nPredicted Labels:")
print(list(y_pred))


# ------------------------------------------------------------
# STEP 11: Calculate Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


# ------------------------------------------------------------
# STEP 12: Display Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=['ham', 'spam']
)

print("\nConfusion Matrix:")
print(cm)


# ------------------------------------------------------------
# STEP 13: Display Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# STEP 14: Test the Model with a New Message
# ------------------------------------------------------------

new_message = [
    'Congratulations! You have won a free cash prize'
]


# Convert the new message into numerical format
new_message_vectorized = vectorizer.transform(new_message)


# Predict whether it is spam or ham
prediction = model.predict(new_message_vectorized)


print("\nNew Message:")
print(new_message[0])

print("Prediction:")
print(prediction[0])


# ------------------------------------------------------------
# STEP 15: Test Another Normal Message
# ------------------------------------------------------------

new_message2 = [
    'Please attend the class tomorrow at 10 AM'
]

new_message2_vectorized = vectorizer.transform(new_message2)

prediction2 = model.predict(new_message2_vectorized)


print("\nNew Message:")
print(new_message2[0])

print("Prediction:")
print(prediction2[0])
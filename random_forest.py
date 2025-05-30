import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("mxmh_survey_results.csv")

df.head(3)

df.shape

df.columns

df.dtypes

df.info()

df.describe()

df.nunique()

"""DATA CLEANING"""

df.isnull().sum()

df.drop(columns=["Permissions"], inplace=True)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Primary streaming service'] = df['Primary streaming service'].fillna(df['Primary streaming service'].mode()[0])

df.dropna(inplace=True)

df.drop_duplicates(inplace=True)

"""FUTURE ENGINEERING

Average mental health score will be calculated and compared for each Fav genre.
Mental health score: Anxiety, Depression, Insomnia, OCD will be average.
"""

genre_scores = df.groupby('Fav genre')['Mental_Health_Score'].mean().sort_values(ascending=False)
print(genre_scores)

plt.figure(figsize=(12,6))
sns.barplot(x=genre_scores.values, y=genre_scores.index, palette='coolwarm')
plt.xlabel("Average Mental Health Score")
plt.ylabel("Favorite Genre")
plt.title("Average Mental Health Score by Favorite Music Genre")
plt.tight_layout()
plt.show()

"""# MODELING

"Can a person's favorite music genre and listening time predict a high mental health score?"



*   Fav genre (kategorik)

*   Hours per day (sayısal)
"""

df['Mental_Health_Score'] = df[['Anxiety', 'Depression', 'Insomnia', 'OCD']].mean(axis=1)

threshold = df['Mental_Health_Score'].median()
df['Mental_Health_Label'] = df['Mental_Health_Score'].apply(lambda x: 1 if x >= threshold else 0)

"""
1.   1 = Higher risk of mental health problems
2.   0 = Lower risk

"""

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Fav_genre_encoded'] = le.fit_transform(df['Fav genre'])

X=df[['Fav_genre_encoded', 'Hours per day']]
y=df['Mental_Health_Label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importances in Random Forest")
plt.show()

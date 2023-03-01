import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the heatwave data
data = pd.read_csv("heatwave_data.csv")

# Split the data into features and target variables
X = data.drop('Heatwave', axis=1)
y = data['Heatwave']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the random forest classifier
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Evaluate the model on the testing data
accuracy = classifier.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Save the model as a pickle file
pickle.dump(classifier, open("model.pickle", "wb"))

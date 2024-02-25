import numpy as np
import pandas as pd
import os
# from sklearn.ensemble import RandomForestClassifier
# import sklearn

for dirname, _, filenames in os.walk('csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Output the train csv
train_data = pd.read_csv("csv/train.csv")
train_data.head()

# Output the test csv
test_data = pd.read_csv("csv/test.csv")
test_data.head()

# Check female survivors
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)


# Check male survivors
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# Look for patterns
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
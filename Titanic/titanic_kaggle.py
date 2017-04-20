import numpy as np
import pandas as pd
from sklearn import preprocessing,neighbors,cross_validation,svm,tree,ensemble
from sklearn.utils import shuffle
import csv





def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))



    return df


# handling the data

df = pd.read_csv('train.csv')

df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)

# print(df.head())

# for test.csv file
df_test = pd.read_csv('test.csv')

ids=df_test['PassengerId']

df_test = df_test.drop(['PassengerId'], 1)
df_test = df_test.drop(['Name'], 1)
df_test.fillna(0, inplace=True)
df_test = handle_non_numerical_data(df_test)
values_for_classifier = np.array(df_test).astype(float)
values_for_classifier = preprocessing.scale(df_test)

# setting features and labels
X = np.array(df.drop(['Survived'], 1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['Survived'])
#
# data for training and testing
df = shuffle(df)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
#
# training the classifier
clf = svm.SVC()

clf.fit(X_train, y_train)




# testing
accuracy = clf.score(X_test, y_test)
print(accuracy)


prediction = clf.predict(np.array(values_for_classifier))

x=np.array(ids)
y=np.array(prediction)

d=[]
for i in range(0,len(x)):
   d.append(str(x[i])+" "+str(y[i]))

print(d)
with open("anwer.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(d)


def fun():
    x=np.array([1,2,3,4,5])
    y=np.array([0,0,0,0,0])
    print(y)
    d=[]
    for i in range(0,len(x)):
        d.append(str(x[i])+str(y[i]))

    print(d)
    with open("anwer.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(d)


# fun()
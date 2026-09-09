import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=sns.load_dataset("titanic")

df.head()

df.columns



df.duplicated()

df.drop_duplicates()

df.dtypes

len(list(df.columns))

numeric=df.select_dtypes(include="number").columns

numeric

categoric=df.drop(numeric,axis=1).columns

categoric

df["sex"].value_counts()

df["who"].value_counts()

sns.histplot(df["who"])

sns.countplot(data=df, x="alive", hue="who")

df.info()

"""deck has more null so remove it

"""

dropeddf=df.drop("deck",axis=1)



print("before dropping",df.shape)
print("after dropping",dropeddf.shape)



"""dropping some more

"""

dropeddf=dropeddf.drop(["embark_town","alive","adult_male","class","who","adult_male"],axis=1)

print("before dropping",df.shape)
print("after dropping",dropeddf.shape)

dropeddf.info()

dropeddf.isnull().sum()

"""age has null vales so lets fill it

"""

sns.boxplot(x=dropeddf["sex"],y=dropeddf["age"])

dropeddf["age"]=dropeddf["age"].fillna(dropeddf["age"].mean())



dropeddf["age"].isnull().sum()

#lets drop embarked null rows not fill but drop only those who hs null

dropeddf.dropna(subset=["embarked"],inplace=True)



"""now label encoding"""

dropeddf.shape

from sklearn.preprocessing import LabelEncoder

objscols=dropeddf.select_dtypes(include="object").columns
objscols

enc=LabelEncoder()

dropeddf[objscols[0]]=enc.fit_transform(dropeddf[objscols[0]])

dropeddf[objscols[1]]=enc.fit_transform(dropeddf[objscols[1]])
#or
# for i in objscols:
#   dropeddf[i]=enc.fit_transform(dropeddf[i])

dropeddf.head()

from sklearn.preprocessing import StandardScaler

# scaler=StandardScaler()
# dropeddf[objscols]=scaler.fit_transform(dropeddf[objscols])



dropeddf.head()

dropeddf.dtypes



"""convert bool to int

"""

boolens=dropeddf.select_dtypes(include="bool").columns
boolens

for i in boolens:
  dropeddf[i]=dropeddf[i].astype(int)



dropeddf.info()

dropeddf.head()

X=dropeddf.drop("survived",axis=1)
Y=dropeddf["survived"]

X.head()

Y.head()

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

from sklearn import metrics

accuracyscore=metrics.accuracy_score(ytest,ypred)

print("the accuracy score is ",accuracyscore)

f1score=metrics.f1_score(ytest,ypred)
precisonscore=metrics.precision_score(ytest,ypred)
recallscore=metrics.recall_score(ytest,ypred)
print("f1score =",f1score)
print("precisonscore =",precisonscore)
print("recallscore =",recallscore)

confmat=metrics.confusion_matrix(ytest,ypred)

print(model.classes_)

confmat

sns.heatmap(confmat,label=model.classes_,annot=True,fmt="d")

sns.heatmap(confmat,label=model.classes_,annot=True)

print(metrics.classification_report(ytest,ypred))

pre=confmat[1][1]/(confmat[1][1]+confmat[0][1])
pre

print("f1score =",f1score)
print("precisonscore =",precisonscore)
print("recallscore =",recallscore)

"""## **lets try another model algorithm**"""

from sklearn.tree import DecisionTreeRegressor

model2=DecisionTreeRegressor()
model2.fit(xtrain,ytrain)

ypred2=model2.predict(xtest)

accuracyscore2=metrics.accuracy_score(ytest,ypred2.astype(int))
precisonscore2=metrics.precision_score(ytest,ypred2.astype(int))
recallscore2=metrics.recall_score(ytest,ypred2.astype(int))

print("accuracyscore2",accuracyscore2)
print("precisonscore2",precisonscore2)
print("recallscore2",recallscore2)

ypred2

print("\t\tlinear regression\t\tdecision tree");
print("accuracy\t",accuracyscore,"\t\t",accuracyscore2);
print("precision\t",precisonscore,"\t\t",precisonscore2);
print("recall\t\t",recallscore,"\t\t",recallscore2);

dff=dropeddf.copy()
dff

scaler=StandardScaler()

dff.head()



dff[["age","fare"]]=scaler.fit_transform(dff[["age","fare"]])

dff.head()

xtrainknn,xtestknn,ytrainknn,ytestknn=train_test_split(dff,Y,test_size=0.2,random_state=42)

from sklearn.neighbors import KNeighborsClassifier

modelknn=KNeighborsClassifier(n_neighbors = 25)
modelknn

modelknn.fit(xtrainknn,ytrainknn)

knnpred=modelknn.predict(xtestknn)

knnaccuracy=metrics.accuracy_score(ytestknn,knnpred)
knnprecision=metrics.precision_score(ytestknn,knnpred)
knnrecall=metrics.recall_score(ytestknn,knnpred)

print("knnaccuracy",knnaccuracy)
print("knnprecision",knnprecision)
print("knnrecall",knnrecall)

sns.heatmap(metrics.confusion_matrix(ytestknn,knnpred),label=modelknn.classes_,annot=True)

print("\t\tlinear regression\t\tdecision tree\t\t\tKNN");
print("accuracy\t",accuracyscore,"\t\t",accuracyscore2,"\t\t",knnaccuracy);
print("precision\t",precisonscore,"\t\t",precisonscore2,"\t\t",knnprecision );
print("recall\t\t",recallscore,"\t\t",recallscore2,"\t\t",knnrecall );



"""# KNN is giving best **results**"""
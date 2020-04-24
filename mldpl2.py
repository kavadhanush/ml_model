
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

datasets=load_iris()
data=datasets.data
output=datasets.target

xtrain,xtest,ytrain,ytest=train_test_split(data,output,test_size=0.3,random_state=10)
model=LogisticRegression()
model.fit(xtrain,ytrain)

a=model.predict(xtest)

pickle.dump(model,open('deploy2.pkl','wb'))

kava=pickle.load(open('deploy2.pkl','rb'))
print(kava.predict([[1.2,0.9,2.1,2.3]]))
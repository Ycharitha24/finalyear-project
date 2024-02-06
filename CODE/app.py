from flask import Flask,render_template,request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pickle
from imblearn.over_sampling import SMOTE


app = Flask(__name__)


x = pd.read_csv(r'Datasets/Book1.csv')

x.head()

x.tail()

x.isnull().sum()

x.dropna(how = 'any',inplace = True)

x.isnull().any().sum()
#
# x.shape

y = x.Output

y.head()

X = x.iloc[:,:10]

X.head()

# sns.countplot(x = X.Security_related_project)
#
# sns.countplot(x = X.Technology_driven_project)
#
# sns.countplot(x = X.Business_driven_project)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 10)

# x_test.shape
#
# y_train.shape

model_1 = DecisionTreeClassifier(max_depth = 1,criterion = 'entropy')

model_1.fit(x_train,y_train)

model_1_pred = model_1.predict(x_test)

# model_1_pred

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,classification_report
# confusion_matrix(y_test,model_1_pred)

model_1.score(x_test,y_test)

precision_score(y_test,model_1_pred)

recall_score(y_test,model_1_pred)

print(classification_report(y_test,model_1_pred))

model_1.score(x_train,y_train)

ex = [[1,1,1,0,1,1,0,1,0,1]]

model_1.predict(ex)
filename = 'Models/Decision Tree.h5'
pickle.dump(model_1, open(filename, 'wb'))


model_2 = RandomForestClassifier(max_depth = 1,n_estimators = 1)

model_2.fit(x_train,y_train)

model_2_pred = model_2.predict(x_test)

# model_2_pred

model_2.score(x_train,y_train)

model_2.score(x_test,y_test)

confusion_matrix(y_test,model_2_pred)

model_2.predict(ex)

precision_score(y_test,model_2_pred)

recall_score(y_test,model_2_pred)
filename = 'Models/RandomForest.h5'
pickle.dump(model_2, open(filename, 'wb'))

from xgboost import XGBClassifier

model_3 = xgb.XGBClassifier()
model_3.fit(x_train,y_train)

model_3_pred = model_3.predict(x_test)
# model_3_pred

confusion_matrix(y_test,model_3_pred)

model_3.score(x_train,y_train)

model_3.score(x_test,y_test)

df_pred=pd.DataFrame(ex[0],index=x_test.columns).transpose()

# df_pred

model_3.predict(df_pred)
filename = 'Models/XGBoost.h5'
pickle.dump(model_3, open(filename, 'wb'))

# model_3.get_booster().feature_names

from sklearn.linear_model import LogisticRegression

model_4 = LogisticRegression(solver='liblinear')

model_4.fit(x_train,y_train)

model_4_pred = model_4.predict(x_test)

# model_4_pred

confusion_matrix(y_test,model_4_pred)

model_4.score(x_test,y_test)

model_4.score(x_train,y_train)
filename = 'Models/Logistic Regression.h5'
pickle.dump(model_4, open(filename, 'wb'))

from sklearn.ensemble import BaggingClassifier


model_5 = BaggingClassifier()

model_5.fit(x_train,y_train)

model_5_pred = model_5.predict(x_test)

# model_5_pred

confusion_matrix(y_test,model_5_pred )

model_5.score(x_test,y_test)

model_5.score(x_train,y_train)
filename = 'Models/Bagging Classifier.h5'
pickle.dump(model_5, open(filename, 'wb'))

accuracy_df = pd.DataFrame({'Model':['Decision Tree','Random Forest','xgboost','Logistic Regression','Bagging Classifier'],
                            'Accuracy' : [model_1.score(x_test,y_test)*100,model_2.score(x_test,y_test)*100,model_3.score(x_test,y_test)*100,model_4.score(x_test,y_test)*100,model_5.score(x_test,y_test)*100]
                           })

# print(accuracy_df)

# sns.barplot(x =accuracy_df.Model,y = accuracy_df.Accuracy)
# plt.xticks(rotation = 'vertical')
# plt.show()


## Class Imbalance Treatment
sm = SMOTE()
x_r, y_r = sm.fit_resample(X, y)
print(x_r.shape, y_r.shape)
print(y_r.value_counts())
## Training and testing after class imbalance treatment
X_train1,X_test1,y_train1,y_test1 = train_test_split(x_r, y_r, test_size=0.3, random_state=10)
from catboost import CatBoostClassifier
def extension():
    global p
    cb = CatBoostClassifier()

    cb.fit(X_train1,y_train1)
    p = cb.predict(X_test1)
    filename= 'Models/catboost.h5'
    pickle.dump(cb,open(filename,'wb'))
    a = print(classification_report(y_test1,p))
    return a
extension()
print(classification_report(y_test,model_1_pred))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/pred',methods =["POST","GET"])
def pred():
    if request.method == 'POST':
        a = int(request.form['f1'])
        b = int(request.form['f2'])
        c = int(request.form['f3'])
        d = int(request.form['f4'])
        e = int(request.form['f5'])
        f = int(request.form['f6'])
        g = int(request.form['f7'])
        h = int(request.form['f8'])
        i = int(request.form['f9'])
        j = int(request.form['f10'])
        input_values = [[int(a),int(b),int(c),int(d),int(e),int(f),int(g),int(h),int(i),int(j)]]
        filename = r"Models/catboost.h5"
        final_model = pickle.load(open(filename, 'rb'))
        result = final_model.predict(input_values)
        print(result)

        return render_template('prediction.html',msg = 'success',pred = result[0])
    return render_template('prediction.html')

@app.route('/graphs')
def graphs():

    return render_template('graphs.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')
if __name__ == '__main__':
    app.run(debug=True)
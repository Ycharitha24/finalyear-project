from app import accuracy_df,y_test,x_test,y_test1,X_test1
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
print(accuracy_df)
plt.figure(figsize=(10,6))
sns.barplot(x = accuracy_df.Model, y = accuracy_df.Accuracy)
plt.xticks(rotation = 'horizontal')
plt.show()
exis_model = r'Models\Decision Tree.h5'
exis_model = pickle.load(open(exis_model, 'rb'))
extension_model = r'Models\catboost.h5'
extension_model = pickle.load(open(extension_model, 'rb'))
exis_pred = exis_model.predict(x_test)
# print(exis_pred)
extension_pred = extension_model.predict(X_test1)
print(classification_report(y_test,exis_pred))
print(classification_report(y_test1,extension_pred))



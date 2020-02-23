from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import f1_score
from ml_models.wrapper_models import DataBinWrapper
from ml_models.linear_model import *

digits = datasets.load_digits()
data = digits['data']
target = digits['target']
X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.2, random_state=0)

data_bin_wrapper = DataBinWrapper(max_bins=10)
data_bin_wrapper.fit(X_train)
X_train = data_bin_wrapper.transform(X_train)
X_test = data_bin_wrapper.transform(X_test)

# 构建特征函数类
feature_func = SimpleFeatureFunction()
feature_func.build_feature_funcs(X_train, y_train)

maxEnt = MaxEnt(feature_func=feature_func)
maxEnt.fit(X_train, y_train)
y = maxEnt.predict(X_test)

print('f1:', f1_score(y_test, y, average='macro'))
print(maxEnt.w)
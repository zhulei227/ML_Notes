from sklearn.datasets import load_boston
import lightgbm as lgb
from ml_models.explain import Shap, TreeShap

dataset = load_boston()
x_data = dataset.data  # 导入所有特征变量
y_data = dataset.target  # 导入目标值（房价）

lgb_train = lgb.Dataset(x_data, y_data.tolist())

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'max_depth': 3,
    'num_leaves': 10,
    'num_iterations': 10,
    'verbose': 0
}

# train
gbm = lgb.train(params, lgb_train, valid_sets=lgb_train)
model_json = gbm.dump_model()
print(gbm.predict(x_data, pred_contrib=True)[0])
shap = Shap(model_json)
print(shap.pred_one_contrib(x_data[0]))
tree_shape = TreeShap(model_json)
print(tree_shape.pred_one_contrib(x_data[0]))

"""
@author: binc
@time: 2018-10-01
@city: 广东财经大学
"""
import pandas as pd
import numpy as np
from scipy import sparse
import gc
import warnings
import lightgbm
import time
import os
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,SelectFromModel

warnings.filterwarnings("ignore")
path = '.'

train_set = pd.read_table(path + '/round1_iflyad_train.txt')
test_set = pd.read_table(path + '/round1_iflyad_test_feature.txt')
data = pd.concat([train_set,test_set],ignore_index=True,axis=0)
data = data.drop('os_name',axis=1)

# 先对所有数据进行缺失值填充，然后转换为0-1变量
data = data.fillna(-1)
data['day'] = data['time'].apply(lambda x: int(time.strftime('%d',time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime('%H', time.localtime(x))))
data['inner_slot_id_1'] = data['inner_slot_id'].apply(lambda x: str(x).split('_')[0])
data['advert_industry_inner_0'] = data['advert_industry_inner'].apply(lambda x: str(x).split('_')[0])
little_count_feature = ['f_channel','adid','orderid','inner_slot_id','app_id','creative_id','orderid','osv']
def get_key(dict_1):
    return [ k for k, v in dict_1.items() if v == 1]
for feat_1 in little_count_feature:
    key = get_key(data[feat_1].value_counts())
    data[feat_1][data[feat_1].isin(key)]  = -1
    
#bool
bool_feature = ['creative_has_deeplink','creative_is_jump','creative_is_download',
               'creative_is_voicead','creative_is_js']
for i in bool_feature:
    data[i] = data[i].astype(int)
# label encoding
advert_feature = ['advert_id','adid','orderid','advert_industry_inner','advert_name',
                  'campaign_id','creative_id','creative_type','creative_tp_dnf','advert_industry_inner_0']
media_feature = ['app_cate_id','f_channel','app_id','inner_slot_id','app_paid','inner_slot_id_1']
content_feature = ['city', 'province', 'nnt', 'devtype','osv','os']
# 待处理 user_tags , make , model
label_feature = advert_feature + media_feature + content_feature
num_feature = ['creative_width','creative_height','hour','day']


label_enc = LabelEncoder()
for label in label_feature:
    data[label] = label_enc.fit_transform(data[label].astype(str))
print("labelencoding is finish")

onehot_feature = label_feature
predict = data[data['click'] == -1].drop('click',axis=1)
predice_click = predict[['instance_id']]
predice_click['predicted_score'] = 0

train_all = data[data['click'] != -1]
train_y = train_all.click.values
train_x = train_all.drop('click',axis = 1)
if os.path.exists(path + '/feature/base_train_csr1.npz') and True:
    base_train_csr = sparse.load_npz(path + '/feature/base_train_csr1.npz').tocsr().astype(bool)
    base_predict_csr = sparse.load_npz(path+ '/feature/base_predict_csr1.npz').tocsr().astype(bool)
else: 
    base_train_csr = sparse.csr_matrix((len(train_x), 0))
    base_predict_csr = sparse.csr_matrix((len(predict), 0))
    enc_onehot = OneHotEncoder()
    for feature in onehot_feature:
        enc_onehot.fit(data[feature].values.reshape(-1,1))
        base_train_csr = sparse.hstack((base_train_csr, enc_onehot.transform(train_x[feature].values.reshape(-1,1))),
                                       'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc_onehot.transform(predict[feature].values.reshape(-1,1))),
                                         'csr', 'bool')
    print('one-hot is ready \n')
    cv = CountVectorizer()
    cv.fit(data['user_tags'].astype(str))
    base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x['user_tags'].astype(str))), 'csr', 'bool')
    base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict['user_tags'].astype(str))), 'csr', 'bool')
    print('count vectory is ready \n')
                                     
    sparse.save_npz(path+'/feature/base_train_csr1.npz', base_train_csr)
    sparse.save_npz(path + '/feature/base_predict_csr1.npz', base_predict_csr)
    
train_csr = sparse.hstack(
        (sparse.csr_matrix(train_x[num_feature]),base_train_csr),'csr').astype('float32')
predict_csr = sparse.hstack(
        (sparse.csr_matrix(predict[num_feature]), base_predict_csr),'csr').astype('float32')
print(train_csr.shape)

select_faeture = SelectPercentile(chi2, percentile= 95)
select_faeture.fit(train_csr, train_y)
train_csr2 = select_faeture.transform(train_csr)
predict_csr2 = select_faeture.transform(predict_csr)
print('feature selection')
print(train_csr2.shape)

time_start = time.time()
lgb_model = lightgbm.LGBMClassifier(
    boosting_type='gbdt', num_leaves=40, reg_alpha=0, reg_lambda=0.1,
    max_depth=-1, n_estimators=2000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.02, random_state=2018, n_jobs=-1
)
param_grid = {'lgb_model__num_leaves': np.linspace(35, 49,15)
              }
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
best_score = []
for i, (train_index, test_index) in enumerate(skf.split(train_x, train_y)):
    print('%s fold \n'%i)
    lgb_model.fit(train_csr2[train_index], train_y[train_index],
                  eval_set = [(train_csr2[train_index], train_y[train_index]),
                              (train_csr2[test_index], train_y[test_index])],
                  early_stopping_rounds=100 )
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    
    test_pre = lgb_model.predict_proba(predict_csr2, n_estimators = lgb_model.best_iteration_)[:,1]
    print('mean: '+str(test_pre.mean)) 
    predice_click['predicted_score'] +=  test_pre

print("best_score mean: "+str(np.mean(best_score)))
predice_click['predicted_score'] = predice_click['predicted_score'] / 5
print("predicted proba : "+ str(predice_click['predicted_score'].mean()))
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')

time_end = time.time()
print('time cost :'+str(time_end - time_start) +'s' )

predice_click.to_csv(path + "/submission/lgb_baseline_%s.csv" % now, index=False)   


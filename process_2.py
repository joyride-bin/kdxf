"""
  @author: binc
  @time: 2018.01.10
  @address:gdufe
  
"""

import pandas as pd
import numpy as np
from scipy import sparse
import gc
import warnings
import lightgbm
import time
import os
import re
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold,KFold,train_test_split,GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest,chi2,SelectPercentile,SelectFromModel
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
path = '.'


train_set = pd.read_table(path + '/round1_iflyad_train.txt')
test_set = pd.read_table(path + '/round1_iflyad_test_feature.txt')
data = pd.concat([train_set,test_set],ignore_index=True,axis=0)
data = data.drop('os_name',axis=1)
data = data.fillna(-1)


# 对make，model进行处理
data['model'] = data['model'].replace('.',-1)
data['model'] = data['model'].replace('0',-1)
data['model'] = data['model'].replace('-',-1)
data['model'] = data['model'].replace('',-1)
data['model'].loc[data['model'].isnull()] = -1
data['model'][data['model'] == -1] = data['make'][data['model'] == -1] 
data['model'].loc[data['model'].isnull()] = -1

regex1 = re.compile(r'%2\d*b')      # 删除%2525....b
regex2 = re.compile(r'%2\d*')       # 删除 %2522...与下面第二行的顺序不能颠倒，否则处理出错
regex3 = re.compile('\(.+?\)')      # 删除括号里面的东西
data['model'] = data['model'].apply(lambda x: regex3.sub(' ' ,str(x).lower()).strip())
data['model'] = data['model'].apply(lambda x: re.sub('%2522', ' ', str(x)).strip())
data['model'] = data['model'].apply(lambda x: regex1.sub(' ' ,str(x).lower()).strip())
data['model'] = data['model'].apply(lambda x: regex2.sub(' ' ,str(x).lower()).strip())
str_split = ['\s', '\+', "_",',']
for i in str_split:
    data['model'] = data['model'].apply(lambda x: re.sub(i,'-',str(x).strip()))
data['model'] = data['model'].replace('apple', 'iphone')
data['model_1'] = data['model'].apply(lambda x: str(x).split('-')[0])

data['make'] = data['make'].apply(lambda x: regex3.sub(' ' ,str(x).lower()).strip())
data['make'] = data['make'].apply(lambda x: re.sub('%2522', ' ', str(x)).strip())
data['make'] = data['make'].apply(lambda x: regex1.sub(' ' ,str(x).lower()).strip())
data['make'] = data['make'].apply(lambda x: regex2.sub(' ' ,str(x).lower()).strip())
str_split2 = ['\s', '\+', "_",',','-','\.']
for i in str_split2:
    data['make']= data['make'].apply(lambda x: re.split(i, str(x))[0])
data['make'] = data['make'].replace('nan', -1)
data['make'][data['make'] == -1] = data['model_1'][data['make'] == -1]


# 特征的清洗与构造
data['day'] = data['time'].apply(lambda x: int(time.strftime('%d', time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime('%H', time.localtime(x))))
data['inner_slot_id_1'] = data['inner_slot_id'].apply(lambda x: str(x).split('_')[0])
data['advert_industry_inner_0'] = data['advert_industry_inner'].apply(lambda x: str(x).split('_')[0])
data['len_user_tags'] = data['user_tags'].apply(lambda x: len(str(x).split(',')))
data['orderid_1'] = data['orderid'].apply(lambda x: str(x)[:5])
ad_appcate_count= data.groupby('adid')['app_cate_id'].agg({'count'}).reset_index()
ad_appcate_count.columns= ['adid', 'appcate_count']
data = pd.merge(data, ad_appcate_count, on= ['adid'], how='left')
data['tags_s'] = data['user_tags'].apply(lambda x : '_'.join(str(x).split(',')))
regex = re.compile('[a-z]+')
data['tags_str'] = data['tags_s'].apply(lambda x: set(re.findall(regex, x)))

data['osv'] = data['osv'].apply(lambda x: re.sub('[a-z]+', '', str(x).lower()).strip())
data['osv'] = data['osv'].apply(lambda x: re.sub('_', '.', str(x).lower()).strip())
#data['os_osv'] = data['os'].astype(str).values + "_" + data['osv'].astype(str).values    #剧毒特征，线上分数狂掉


# 强特，加了之后线上log_loss从0.42457涨到0.42375 
data['ad_area'] = data['creative_width']*data['creative_height']
data['model_area'] = data['model'].astype(str).values + '_' + data['ad_area'].astype(str).values     

# 处理长尾数据
little_count_feature = ['f_channel','adid','inner_slot_id','app_id','creative_id','orderid']
def get_key(dict_1):
    return [ k for k, v in dict_1.items() if v == 1]
for feat_1 in little_count_feature:
    key = get_key(data[feat_1].value_counts())
    data[feat_1][data[feat_1].isin(key)]  = -1

#bool
bool_feature = ['creative_has_deeplink','creative_is_jump','creative_is_download']
for i in bool_feature:
    data[i] = data[i].astype(int)


advert_feature = ['advert_id','adid','orderid','advert_industry_inner','advert_name',
                  'campaign_id','creative_id','creative_type','creative_tp_dnf','advert_industry_inner_0']
media_feature = ['app_cate_id','f_channel','app_id','inner_slot_id','inner_slot_id_1']
content_feature = ['city', 'province', 'nnt', 'devtype','os','osv', 'make','model_area']
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

#稀疏矩阵
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

# 多值特征处理
cv = CountVectorizer()
cv.fit(data['user_tags'].astype(str))
cv_features = ['user_tags']
for cv_feat in cv_features:
    cv_train =  cv.transform(train_x[cv_feat].astype(str))
    cv_predcit = cv.transform(predict[cv_feat].astype(str))
    base_train_csr = sparse.hstack((base_train_csr,cv_train), 'csr', 'bool')
    base_predict_csr = sparse.hstack((base_predict_csr,cv_predcit), 'csr', 'bool')
print('count vectory is ready \n')

train_csr = sparse.hstack(
        (sparse.csr_matrix(train_x[num_feature]),base_train_csr),'csr').astype('float32')
predict_csr = sparse.hstack(
        (sparse.csr_matrix(predict[num_feature]), base_predict_csr),'csr').astype('float32')
print(train_csr.shape)

# 利用卡方检验来进行特征筛选
select_faeture = SelectPercentile(chi2, percentile= 95)
select_faeture.fit(train_csr, train_y)
train_csr2 = select_faeture.transform(train_csr)
predict_csr2 = select_faeture.transform(predict_csr)
print('feature selection')
print(train_csr2.shape)


time_start = time.time()
lgb_model = lightgbm.LGBMClassifier(
    boosting_type='gbdt', num_leaves=48, reg_alpha=0, reg_lambda=0.1,max_bin=425,
    max_depth=-1, n_estimators=5000, objective='binary',subsample_for_bin=50000,
    subsample=0.8,colsample_bytree=0.8,subsample_freq=1, min_child_weight=5, min_child_samples=10, 
    learning_rate=0.02, random_state=2018, n_jobs=-1
)

# 做五折交叉 得到平均预测
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
best_score = []
for i, (train_index, test_index) in enumerate(skf.split(train_csr_res, train_y)):
    print('%s fold \n'%i)
    lgb_model.fit(train_csr_res[train_index], train_y[train_index],
                  eval_set = [(train_csr_res[train_index], train_y[train_index]),
                              (train_csr_res[test_index], train_y[test_index])],
                  early_stopping_rounds=100 )
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    
    test_pre = lgb_model.predict_proba(test_csr_res, n_estimators = lgb_model.best_iteration_)[:,1]
    print('mean: '+str(test_pre.mean)) 
    predice_click['predicted_score'] +=  test_pre

print("best_score mean: "+str(np.mean(best_score)))
predice_click['predicted_score'] = predice_click['predicted_score'] / 5
print("predicted proba : "+ str( predice_click['predicted_score'].mean()))
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')

time_end = time.time()
print('time cost :'+str((time_end - time_start)/60) +'min' )
predice_click.to_csv(path + "/submission/lgb_model_area%s.csv" % now, index=False)   

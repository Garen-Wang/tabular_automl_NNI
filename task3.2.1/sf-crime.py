import sys

sys.path.append('../')
import nni
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import logging

from fe_util import *
from model import *

logger = logging.getLogger('sf-crime')
train_file_name = '../data/sf-crime/train.csv'
test_file_name = '../data/sf-crime/test.csv'
target_name = 'Category'
id_index = 'Id'
train = pd.read_csv(train_file_name)
test = pd.read_csv(test_file_name)


def fillna(data):
    data = data[data[target_name].notnull()]
    if data[target_name].dtypes == object:
        target_encoder = LabelEncoder()
        data[target_name] = target_encoder.fit_transform(data[target_name])
        target_columns = target_encoder.classes_
    else:
        target_columns = []
    data['Dates'] = data['Dates'].map(lambda x: pd.to_datetime(x))
    data['Year'] = data['Dates'].map(lambda x: x.year)
    data['Month'] = data['Dates'].map(lambda x: x.month)
    data['Day'] = data['Dates'].map(lambda x: x.day)
    data['Hour'] = data['Dates'].map(lambda x: x.hour)
    dayofweek_dict = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 6,
        'Sunday': 7
    }
    data['DayOfWeek'] = data['DayOfWeek'].map(lambda x: dayofweek_dict[x])
    data['PdDistrict'] = LabelEncoder().fit_transform(data['PdDistrict'])
    data['Quarter'] = data['Dates'].map(lambda x: x.quarter)

    def getHourZone(x):
        if x <= 7:
            return 1
        elif x <= 11:
            return 2
        elif x <= 13:
            return 3
        elif x <= 16:
            return 4
        elif x <= 18:
            return 5
        else:
            return 6

    data['HourZone'] = data['Hour'].map(lambda x: getHourZone(x))
    # data[data['Y'] == 90]
    data['HasAddressNum'] = data['Address'].map(lambda x: 1 if 'Block' in x else 0)
    data['AddressNum'] = data['Address'].map(lambda x: x.split(' ')[0])

    def isint(x):
        try:
            int(x)
            return True
        except ValueError:
            pass
        return False

    data['AddressNum'] = data['AddressNum'].map(lambda x: int(x) if isint(x) else -1)
    address_nums = data['AddressNum'].unique()
    for address_num in address_nums:
        if address_num == -1:
            data['AddressNum-' + 'None'] = data['AddressNum'].map(lambda x: 1 if x == -1 else 0)
        else:
            data['AddressNum-' + str(address_num)] = data['AddressNum'].map(lambda x: 1 if x == address_num else 0)

    def getLocation(x):
        if 'Block' in x:
            locations.add(x.split('of')[-1].strip())
            return [x.split('of')[-1].strip()]
        else:
            locations.add(x.split('/')[0].strip())
            locations.add(x.split('/')[1].strip())
            return [x.split('/')[0].strip(), x.split('/')[1].strip()]
    locations = set()
    data['Locations'] = data['Address'].apply(getLocation)
    locations = list(locations)
    for location in locations:
        if len(location.split(' ')[-1]) != 2:
            locations.remove(location)
    suffixes = set()
    for location in locations:
        suffixes.add(location.split(' ')[-1])
    suffixes = list(suffixes)

    def hasSuffix(li, suffix):
        for x in li:
            if x.split(' ')[-1] == suffix:
                return 1
        return 0
    for suffix in suffixes:
        data['suffix-' + suffix] = data['Locations'].map(lambda x: hasSuffix(x, suffix))

    def hasOthers(li):
        for x in li:
            if x.split(' ')[-1] not in suffixes:
                return 1
        return 0
    data['suffix-OTHERS'] = data['Locations'].map(hasOthers)

    data.drop(['Dates', 'AddressNum', 'Descript', 'Resolution', 'Locations', 'Address'], axis=1, inplace=True)
    return data


def init():
    global train, test
    train = fillna(train)
    print(train.columns)
    print(train.info())
    print(train.isnull().sum().sum())


def lightgbm_train(max_epoch=1000, min_data=200):
    # min_data: 1, 3, 9, 97561, 292683
    global train, target_name
    params_lgb = {
        'objective': 'multiclass',
        'metric': 'multiclass',
        'verbose': -1,
        'seed': 1024,
        'num_threads': 4,
        'num_leaves': 64,
        'learning_rate': 0.05,
        'min_data': min_data,
        'bagging_fraction': 0.5,
        'feature_fraction': 0.5,
        'max_depth': -1
    }
    X_train, X_valid, y_train, y_valid = train_test_split(train.drop(target_name, axis=1), train[target_name].values, 0.15, random_state=2077)
    lgb_train = lgb.Dataset(X_train, X_valid)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    classifier = lgb.train(params=params_lgb, train_set=lgb_train, valid_sets=lgb_valid, valid_names='eval', verbose_eval=50, early_stopping_rounds=100, num_boost_round=max_epoch)
    feature_names = [feature_name for feature_name in train.columns if feature_name != target_name]
    feature_importance = get_fea_importance(classifier, feature_names)
    print('best score: %.4f' % classifier.best_score)
    from sklearn.metrics import log_loss
    y_pred = classifier.predict(X_valid, num_iteration=classifier.best_iteration)
    val_loss = log_loss(y_valid, y_pred)
    return val_loss, feature_importance


def main():
    global train, test
    nni_params = nni.get_next_parameter()
    logger.info('NNI params: \n', nni_params)
    if 'sample_feature' in nni_params.keys():
        sample_col = nni_params['sample_feature']
    else:
        sample_col = []
    train = name2feature(train, sample_col, target_name)
    print(train.columns)

    val_score, feature_importance = lightgbm_train(min_data=9)
    # still don't know how to set ascending=True
    nni.report_final_result({
        'default': val_score,
        'feature_importance': feature_importance
    })
    print(feature_importance)


if __name__ == '__main__':
    init()
    main()

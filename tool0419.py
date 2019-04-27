# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:09:41 2019

@author: LXW
"""
import re
import time
import operator
import numpy as np
import pandas as pd
import pickle
import pymongo
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from tpot import TPOTClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# 加载静态模型
def load_model(model_name):
    '''
    :param model_name: 静态模型的文件名
    :return: 加载后的模型
    '''
    model = joblib.load(model_name)
    return model


#直接从数据库里获取特征
def get_data_feature(mogo_url, database, collection_name):
    """
    :param mogo_url: Mongodb的url地址
    :param database: 数据库名
    :param collection_name: 名
    :return: 流量特征
    """
    mongo_url = mogo_url  # '172.18.130.22:26000'
    client = pymongo.MongoClient(mongo_url)

    DATABASE = database  # 'ion'
    db = client[DATABASE]
    COLLECTION = collection_name  # 'sessionFeatureTest0'
    collection = db[COLLECTION]
    flows = collection.find({}, {'isIdentify': 0})
    cnt = flows.count()
    data_feature = []
    for flow in flows:
        # data_feature.append(flow)
        feature = flow['features'].split(',')
        feature_int = [float(x) for x in feature]
        feature_final = feature_int + [flow['key'], flow['ip']]
        data_feature.append(feature_final)
    column = ['proto', 'pktInterval_max', 'pktInterval_min', 'pktInterval_sum', 'pktInterval_med',
              'pktInterval_var', 'pktInterval_std', 'pktInterval_avg', 'pktInterval_upPrecent',
              'pktInterval_downPrecent', 'pktInterval_entropy',
              'pktSize_max', 'pktSize_min', 'pktSize_sum', 'pktSize_med', 'pktSize_var', 'pktSize_std',
              'pktSize_avg', 'pktSize_upPrecent', 'pktSize_downPrecent', 'pktSize_entropy',
              'rxPktInterval_max', 'rxPktInterval_min', 'rxPktInterval_sum', 'rxPktInterval_med',
              'rxPktInterval_var', 'rxPktInterval_std', 'rxPktInterval_avg', 'rxPktInterval_upPrecent',
              'rxPktInterval_downPrecent', 'rxPktInterval_entropy',
              'rxPktSize_max', 'rxPktSize_min', 'rxPktSize_sum', 'rxPktSize_med', 'rxPktSize_var', 'rxPktSize_std',
              'rxPktSize_avg', 'rxPktSize_upPrecent', 'rxPktSize_downPrecent', 'rxPktSize_entropy',
              'txPktInterval_max', 'txPktInterval_min', 'txPktInterval_sum', 'txPktInterval_med',
              'txPktInterval_var', 'txPktInterval_std', 'txPktInterval_avg', 'txPktInterval_upPrecent',
              'txPktInterval_downPrecent', 'txPktInterval_entropy',
              'txPktSize_max', 'txPktSize_min', 'txPktSize_sum', 'txPktSize_med', 'txPktSize_var', 'txPktSize_std',
              'txPktSize_avg', 'txPktSize_upPrecent', 'txPktSize_downPrecent', 'txPktSize_entropy', 'sizeRatio',
              'packets', 'packetsA', 'packetsB', 'packetsRatio', 'key', 'ip']
    data_feature_frame = pd.DataFrame(data_feature, columns=column)
    return data_feature_frame


# 从数据库里读取流的报文特征
def get_data(mogo_url, database, collection_name):
    """
    :param mogo_url: Mongodb的url地址
    :param database: 数据库名
    :param collection_name: 表名
    :return:获取网络流的报文字段
    """

    print('开始获取数据')
    mongo_url = mogo_url  # '172.18.130.22:26000'
    client = pymongo.MongoClient(mongo_url)

    DATABASE = database  # 'ion'
    db = client[DATABASE]
    COLLECTION = collection_name  # 'packetTest3_2019-03-04'
    collection = db[COLLECTION]

    flow = collection.distinct('key')
    print(len(flow))
    print(flow)
    data = []
    sum = 0
    for x in flow:
        sum = sum + 1
        print(sum)
        if sum < len(flow) + 1:
            flowOne = collection.find({'key': x}).sort('timeStamp')

            endTime = flowOne[0]['timeStamp'] + 7200
            lst=collection.find({'timeStamp': {"$lte": endTime}, 'key': x}).sort('timeStamp')[:10000]
            #data.append(collection.find({'timeStamp': {"$lte": endTime}, 'key': x}).sort('timeStamp'))

            data.append(lst)
        else:
            break
    print('获取流信息')
    protocolAll = []  # 使用的协议
    packetSizeAll = []  # 包大小
    packetIntervalAll = []  # 包时间间隔
    rxPacketSizeAll = []
    rxPacketIntervalAll = []
    txPacketSizeAll = []
    txPacketIntervalAll = []
    for i in range(len(data)):
        protocol = data[i][0]['protocol']
        direction = []
        packetSize = []
        packetInterval = []
        rxPacketSize = []
        rxPacketInterval = []
        txPacketSize = []
        txPacketInterval = []
        for y in data[i]:
            packetSize.append(y['pktSize'])
            packetInterval.append(y['pktInterval'])
            rxPacketSize.append(y['rxPktSize'])
            rxPacketInterval.append(y['rxPktInterval'])
            txPacketSize.append(y['txPktSize'])
            txPacketInterval.append(y['txPktInterval'])
            direction.append(y['direction'])
        protocolAll.append(protocol)
        packetSizeAll.append(packetSize)
        packetIntervalAll.append(packetInterval)
        rxPacketSizeAll.append(rxPacketSize)
        rxPacketIntervalAll.append(rxPacketInterval)
        txPacketSizeAll.append(txPacketSize)
        txPacketIntervalAll.append(txPacketInterval)
    print('获取数据成功')
    return flow, protocolAll, packetSizeAll, packetIntervalAll, rxPacketSizeAll, rxPacketIntervalAll, \
           txPacketSizeAll, txPacketIntervalAll


def RemoveNum(a, data):
    result = [x for x in data if x != a]
    return result


# 提取某个字段的数学统计特征
def IoTFeatureExtraction(date):
    '''
    :param date:一个列表序列
    :return: 该列表的最大值、最小值，和、中位数、方差、标准差、平均值、上四分位、下四分位和信息熵
    '''
    if date:
        maximum = max(date)
        minium = min(date)
        summation = sum(date)
        med = np.median(date)
        variance = np.var(date)
        standardDeviation = np.std(date)
        avg = np.mean(date)
        upPrecent = np.percentile(date, 25)
        downPrecent = np.percentile(date, 75)
        date_value = list(set(date))
        entropy = 0
        for i in date_value:
            p = date.count(i) / len(date)
            entropy += -1 * p * np.log(p)
    else:
        maximum = 0
        minium = 0
        summation = 0
        med = 0
        variance = 0
        standardDeviation = 0
        avg = 0
        upPrecent = 0
        downPrecent = 0
        entropy = 0
    return [maximum, minium, summation, med, variance, standardDeviation, avg, upPrecent, downPrecent, entropy]


# 获取流量特征
def data_process(data, iot_dict):
    """
    :param data: 包含流、协议、包的大小、包的时长的列表
    :param iot_dict: ip所属类别
    :return:提取后的流的特征向量
    """
    print('开始处理数据')
    flow = data[0]
    protocolAll = data[1]
    packetSizeAll = data[2]
    packetIntervalAll = data[3]
    rxPacketSizeAll = data[4]
    rxPacketIntervalAll = data[5]
    txPacketSizeAll = data[6]
    txPacketIntervalAll = data[7]
    result = []
    length = len(flow)
    for i in range(length):
        print('正在处理第{}条，一共{}条'.format(i, length))
        quintuple = flow[i]
        ip_10_0 = "10\.0\.((?:(?:25[0-5]|2[0-4]\\d|((1\\d{2})|([1-9]?\\d)))\\.){1}(?:25[0-5]|2[0-4]\\d|((1\\d{2})|([1-9]?\\d))))"
        flow_split = flow[i].split(',')
        ip1 = flow_split[0]
        ip2 = flow_split[2]
        """
        ip = ip1
        if ip1 not in iot_dict and (ip2 in iot_dict or (re.match(ip_10_0, ip2) is not None)):
            ip = ip2
        """
        if ip1 in iot_dict:
            ip = ip1
        elif ip2 in iot_dict:
            ip = ip2
        elif re.match(ip_10_0, ip1) is not None:
            ip = ip1
        elif re.match(ip_10_0, ip2) is not None:
            ip = ip2
        else:
            ip = ip1
        # 协议
        protocolE = protocolAll[i]
        # 全部包时间特征
        pktIntervalE = IoTFeatureExtraction(packetIntervalAll[i])
        # 全部包大小特征
        pktSizeE = IoTFeatureExtraction(packetSizeAll[i])
        # rx包时间特征
        rxPktIntervalE = IoTFeatureExtraction(RemoveNum(0, rxPacketIntervalAll[i]))
        # rx包大小特征
        rxPktSizeE = IoTFeatureExtraction(RemoveNum(0, rxPacketSizeAll[i]))
        # tx包时间特征
        txPktIntervalE = IoTFeatureExtraction(RemoveNum(0, txPacketIntervalAll[i]))
        # tx包大小特征
        txPktSizeE = IoTFeatureExtraction(RemoveNum(0, txPacketSizeAll[i]))
        # rx tx大小比
        sizeRatio = [(rxPktSizeE[2] + 0.01) / (txPktSizeE[2] + 0.01)]
        # 全部包个数
        packets = [len(packetSizeAll[i])]
        # rx包个数
        packetsA = [len(RemoveNum(0, rxPacketSizeAll[i]))]
        # tx包个数
        packetsB = [len(RemoveNum(0, txPacketSizeAll[i]))]
        # rx tx包个数比
        packetsRatio = [(packetsA[0] + 0.1) / (packetsB[0] + 0.1)]
        result.append(
            [protocolE] + pktIntervalE + pktSizeE + rxPktIntervalE + rxPktSizeE + txPktIntervalE + txPktSizeE +
            sizeRatio + packets + packetsA + packetsB + packetsRatio + [quintuple, ip])
        column = ['proto', 'pktInterval_max', 'pktInterval_min', 'pktInterval_sum', 'pktInterval_med',
                  'pktInterval_var', 'pktInterval_std', 'pktInterval_avg', 'pktInterval_upPrecent',
                  'pktInterval_downPrecent', 'pktInterval_entropy',
                  'pktSize_max', 'pktSize_min', 'pktSize_sum', 'pktSize_med', 'pktSize_var', 'pktSize_std',
                  'pktSize_avg', 'pktSize_upPrecent', 'pktSize_downPrecent', 'pktSize_entropy',
                  'rxPktInterval_max', 'rxPktInterval_min', 'rxPktInterval_sum', 'rxPktInterval_med',
                  'rxPktInterval_var', 'rxPktInterval_std', 'rxPktInterval_avg', 'rxPktInterval_upPrecent',
                  'rxPktInterval_downPrecent', 'rxPktInterval_entropy',
                  'rxPktSize_max', 'rxPktSize_min', 'rxPktSize_sum', 'rxPktSize_med', 'rxPktSize_var', 'rxPktSize_std',
                  'rxPktSize_avg', 'rxPktSize_upPrecent', 'rxPktSize_downPrecent', 'rxPktSize_entropy',
                  'txPktInterval_max', 'txPktInterval_min', 'txPktInterval_sum', 'txPktInterval_med',
                  'txPktInterval_var', 'txPktInterval_std', 'txPktInterval_avg', 'txPktInterval_upPrecent',
                  'txPktInterval_downPrecent', 'txPktInterval_entropy',
                  'txPktSize_max', 'txPktSize_min', 'txPktSize_sum', 'txPktSize_med', 'txPktSize_var', 'txPktSize_std',
                  'txPktSize_avg', 'txPktSize_upPrecent', 'txPktSize_downPrecent', 'txPktSize_entropy', 'sizeRatio',
                  'packets', 'packetsA', 'packetsB', 'packetsRatio', 'key', 'ip']
    print('数据处理完毕')
    data_frame = pd.DataFrame(result, columns=column)
    return data_frame


def ip_class_info(ip_info_name):
    '''
    :param ip_info_name: ip信息文件名
    :return: ip所属类别信息
    '''
    with open(ip_info_name, 'rb') as f:
        iot_dict = pickle.load(f)
    ip_class_name = iot_dict['ip_class']
    iot_dict.pop('ip_class')
    ip_calss_index = list(set(iot_dict.values()))
    assert len(ip_calss_index) == len(ip_class_name), '类别个数不一致'
    ip_class_name_index = [idx for idx in range(len(ip_class_name))]
    ip_calss_index.sort()
    assert operator.eq(ip_class_name_index, ip_calss_index), '类别不连续'
    iot_dict['ip_class'] = ip_class_name
    return iot_dict


def pred_test_data(model, data_frame, iot_dict, threshold):
    '''
    :param model: 静态模型
    :param data_test: 测试数据
    :param threshold: 阈值
    :param iot_dict: 每个ip所属的类别
    :return:
    '''
    """
    model.classes_ = model.classes_ - 1
    data_test = data_frame.iloc[:, :-2]
    pred_prob = pd.DataFrame(model.predict_proba(data_test), columns=model.classes_)
    greater_threshold_idx = pred_prob.max(axis=1) >= threshold
    pred_class = pred_prob[greater_threshold_idx].idxmax(axis=1)
    smaller_threshold_idx = pred_prob.max(axis=1) < threshold
    smaller_threshold_data = data_frame[smaller_threshold_idx]

    greater_threshold_data = data_frame[greater_threshold_idx]
    greater_threshold_data['label'] = greater_threshold_data['ip'].map(lambda x: iot_dict.get(x, -1))
    greater_threshold_data['pred'] = pred_class
    greater_threshold_data_iniot_dict = greater_threshold_data[greater_threshold_data['label'] != -1]
    learn_data = greater_threshold_data_iniot_dict[
    greater_threshold_data_iniot_dict['label'] == greater_threshold_data_iniot_dict['pred']]
    error_data = greater_threshold_data_iniot_dict[
    greater_threshold_data_iniot_dict['label'] != greater_threshold_data_iniot_dict['pred']]
    to_pred_data = greater_threshold_data[greater_threshold_data['label'] == -1]
    learn_data.drop(columns=['pred', 'label'], axis=1, inplace=True)
    error_data.drop(columns=['pred', 'label'], axis=1, inplace=True)
    to_pred_data.drop(columns=['pred', 'label'], axis=1, inplace=True)
    smaller_threshold_data['label'] = smaller_threshold_data['ip'].map(lambda x: iot_dict.get(x, -1))
    smaller_threshold_learn_data = smaller_threshold_data[smaller_threshold_data['label'] != -1]
    smaller_threshold_to_pred_data = smaller_threshold_data[smaller_threshold_data['label'] == -1]
    smaller_threshold_learn_data.drop(columns=['label'], axis=1, inplace=True)
    smaller_threshold_to_pred_data.drop(columns=['label'], axis=1, inplace=True)
    learn_data = pd.concat([learn_data, smaller_threshold_learn_data])
    to_pred_data = pd.concat([to_pred_data, smaller_threshold_to_pred_data])
    return learn_data, to_pred_data, error_data
    """
    isin_iot_dict = data_frame['ip'].map(lambda x: iot_dict.get(x, -1))
    index = isin_iot_dict == -1
    to_pred_data = data_frame[isin_iot_dict == -1]
    isin_iot_dict_data = data_frame[isin_iot_dict != -1]
    isin_iot_dict_data.reset_index(drop=True, inplace=True)
    model.classes_ = model.classes_ - 1
    data_test = isin_iot_dict_data.iloc[:, :-2]
    pred_prob = pd.DataFrame(model.predict_proba(data_test), columns=model.classes_)
    greater_threshold_idx = pred_prob.max(axis=1) >= threshold
    pred_class = pred_prob[greater_threshold_idx].idxmax(axis=1)
    # greater_threshold_num = len(pred_class)
    smaller_threshold_idx = pred_prob.max(axis=1) < threshold
    # smaller_threshold_idx = np.where(smaller_threshold_idx ==True)[0]
    smaller_threshold_num = len(smaller_threshold_idx)
    smaller_threshold_data = isin_iot_dict_data[smaller_threshold_idx]
    model_class_lst = list(model.classes_)
    smaller_threshold_data_label = smaller_threshold_data['ip'].map(lambda x: iot_dict.get(x, -1))
    smaller_threshold_data_label_isin_model = smaller_threshold_data_label.map(lambda x: x in model_class_lst)
    num = sum(smaller_threshold_data_label_isin_model)
    if num > 0:
        smaller_threshold_data_label_isin_model_data = smaller_threshold_data[smaller_threshold_data_label_isin_model]

    greater_threshold_data = isin_iot_dict_data[greater_threshold_idx]
    greater_threshold_data['label'] = greater_threshold_data['ip'].map(lambda x: iot_dict.get(x, -1))
    greater_threshold_data['pred'] = pred_class
    greater_threshold_data_iniot_dict = greater_threshold_data[greater_threshold_data['label'] != -1]
    if len(greater_threshold_data_iniot_dict) != len(greater_threshold_data):
        raise Exception('异常')
    learn_data = greater_threshold_data_iniot_dict[
    greater_threshold_data_iniot_dict['label'] == greater_threshold_data_iniot_dict['pred']]
    error_data = greater_threshold_data_iniot_dict[
    greater_threshold_data_iniot_dict['label'] != greater_threshold_data_iniot_dict['pred']]
    learn_data.drop(columns=['pred', 'label'], axis=1, inplace=True)
    error_data.drop(columns=['pred', 'label'], axis=1, inplace=True)

    learn_data = pd.concat([learn_data, smaller_threshold_data])
    return learn_data, to_pred_data, error_data


def learn_data_is_enough(learn_data, iot_dict, num_per_class):
    '''
    :param learn_data: 学习集
    :param iot_dict: 每个ip对应的类别
    ：num_per_class: 每类设备需要的流数
    :return: 学习集是否足够
    '''
    learn_data['label'] = learn_data.iloc[:, -1].map(lambda x: iot_dict.get(x, -1))
    classes_info = learn_data.groupby(learn_data['label']).count()
    classes_num = classes_info.iloc[:, 1]
    flag = True
    for idx in range(len(classes_num)):
        tmp_num = classes_num.iloc[idx]
        if tmp_num < num_per_class:
            print('第{0}类的流样本数量为{1}，还需要{2}个样本'.format(idx, tmp_num, num_per_class - tmp_num))
            flag = False
    return flag


def merger_data(original_data, learn_data, sample_num):
    '''
    :param original_data: 原始的学习集
    :param learn_data: 学习集
    :param iot_dict: ip映射到类别的字典
    :param sample_num: 每类采集的流数
    :return: 用于监督模型的训练集
    '''
    data = pd.concat([original_data, learn_data], axis=0)
    # data['label'] = data['ip'].map(lambda x: iot_dict.get(x, -1))
    label_list = list(data['label'].unique())
    train_data = pd.DataFrame(columns=original_data.columns)
    for class_idx in label_list:
        if class_idx == -1:
            continue
        tmp_data = data[data['label'] == class_idx]
        if len(tmp_data) >= sample_num and (class_idx != -1):  # -1表示类别不确定
            tmp_data = tmp_data.sample(sample_num)
        train_data = pd.concat([train_data, tmp_data], axis=0)
    return train_data


def train_model(train_data, precision=0.9):
    '''
    :param train_data: 训练数据集
    :return: 训练后的静态模型
    '''
    feature = train_data.iloc[:, :-3].astype(np.float)
    label = train_data.iloc[:, -1].astype(np.int)
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3, random_state=6)
    rf0 = RandomForestClassifier(n_estimators=1000, random_state=17)
    bst = XGBClassifier()
    tpot = TPOTClassifier(generations=3, population_size=20, verbosity=2)
    tpot.fit(X_train, y_train)
    bst.fit(X_train, y_train)
    rf0.fit(X_train, y_train)
    result_rf = rf0.predict(X_test)
    result_bst = bst.predict(X_test)
    acc = accuracy_score(y_test, result_bst)
    if acc < precision:
        print('训练过程准确率小于{}, 建议检查数据质量'.format(precision))
    print('acc={}'.format(accuracy_score(y_test, result_rf)))
    print('Classification report for classifier:\n{}'.format(classification_report(y_test, result_rf)))
    print("Confusion matrix:\n{}".format(confusion_matrix(y_test, result_rf,
                                                          labels=rf0.classes_)))
    return rf0, bst, tpot


def pred_topred_data(model, pred_data, threshold=0.8):
    '''
    :param model: 新静态模型
    :param pred_data: 预测的数据
    :return:每个iP的预测类别
    '''
    pred_prob = pd.DataFrame(model.predict_proba(pred_data.iloc[:, :-2].astype('float'))) #,  columns=model.classes)
    pred_label = pred_prob.idxmax(axis=1)
    class_num = pred_prob.shape[1]
    ip_num = len(pred_data['ip'].unique())
    ip_list = pred_data['ip'].unique()
    pred_ip_info = pd.DataFrame(np.zeros([ip_num, class_num + 1], np.int8), index=ip_list)
    for idx in range(len(pred_data)):
        tmp_ip = pred_data.iloc[idx, -1]
        tmp_label = pred_label[idx]
        if pred_prob.iloc[idx, tmp_label] >= threshold:
            pred_ip_info.loc[tmp_ip, tmp_label] += 1
        else:
            pred_ip_info.loc[tmp_ip, class_num] += 1
    pred_ip_info['pred'] = pred_ip_info.idxmax(axis=1)
    return pred_ip_info['pred']


if __name__ == "__main__":

    start = time.time()
    # 1.加载ip信息表
    iot_dict = ip_class_info('ip_class.pkl')

    # 2.加载原始数据集,判断原始数据集每个ip是否都有标签

    origin_packet_data = get_data('172.18.130.22:26000', 'ion', 'packetTest0')
    origin_flow_feature = data_process(origin_packet_data, iot_dict)

    # origin_flow_feature = pd.read_csv('origin_flow_feature.csv') #本地文件直接读
    # 从数据库里直接读取特征
    # origin_flow_feature = get_data_feature('172.18.130.22:26000', 'ion', 'sessionFeatureTest0')
    origin_flow_feature['label'] = origin_flow_feature['ip'].map(lambda x: iot_dict.get(x, -1))

    indice_nolabel = np.where(origin_flow_feature['label'] == -1)[0]
    if len(indice_nolabel) > 0:
        print('原始数据集部分没有标签，索引为{}'.format(list(indice_nolabel)))
        # exit()
        origin_flow_feature = origin_flow_feature[origin_flow_feature['label'] != -1]
        origin_flow_feature.reset_index(drop=True, inplace=True)

    # 3.加载原始的静态模型
    origin_model = load_model('iot_v1.joblib')

    # 4.加载测试数据

    test_data_packet = get_data('172.18.130.22:26000', 'ion', 'packetTest4')
    test_data_flow = data_process(test_data_packet, iot_dict)
    # 从本地文件读取数据
    # test_data_flow = pd.read_csv('test_data_flow.csv')

    # 从数据库里直接读取特征
    # test_data_flow = get_data_feature('172.18.130.22:26000', 'ion', 'sessionFeatureTest4')
    # 5.原始静态模型对测试集进行预测，得到学习集，待测数据和预测错的数据集
    learn_data, to_pred_data, error_data = pred_test_data(origin_model, test_data_flow, iot_dict, 0.8)
    # learn_data = pd.read_csv('learn_data.csv')
    # to_pred_data  =pd.read_csv('to_pred_data.csv')
    to_pred_data = pd.concat([to_pred_data, error_data])
    # 6.判断学习集是否足够
    flag = learn_data_is_enough(learn_data, iot_dict, 500)

    # 7.合并学习集与原始学习集，随机采样获得训练集
    if not flag:
        print('数据集不够，请补充数据集')
        # exit()
    train_data = merger_data(origin_flow_feature, learn_data, 2000) #1000

    # 8.训练新的静态模型
    rf, bst, tpot = train_model(train_data, precision=0.9)

    # 9.对数据集进行预测，输出每个ip的类别
    result_rf = pred_topred_data(rf, to_pred_data, 0.7) #0.7
    result_bst = pred_topred_data(bst, to_pred_data, 0.7)
    result_tpot = pred_topred_data(tpot, to_pred_data, 0.7)

    test = get_data('172.18.130.22:26000', 'ion', 'packetTest1')
    test_data = data_process(test, iot_dict)
    res = pred_topred_data(rf, test_data, 0.7)
    end = time.time()
    print('运行完毕，使用时间{}秒！'.format(end - start))






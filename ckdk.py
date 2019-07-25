import time
import datetime
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from statsmodels.tsa.seasonal import seasonal_decompose

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
begin = time.time()
# 读取数据
data = pd.read_csv('ckdk_data0725.csv', header=0)
# data = pd.read_csv('kpi_type2.csv', index_col=0, header=0)

data['date'] = pd.to_datetime(data['ts'])
for i in range(len(data)):
	tmp_datetime = data['date'][i]
	minutes = tmp_datetime.minute
	if minutes > 0:
		tmp = tmp_datetime - datetime.timedelta(minutes=minutes)
		data['date'][i] = tmp
data.drop('ts', axis=1, inplace=True)
print('the shape of raw data is: ', data.shape)
helper = pd.DataFrame({'date': pd.date_range(data['date'].min(), data['date'].max(), freq='H')})
print('the shape of helper is: ', helper.shape)
data_merge = pd.merge(data, helper, on='date', how='right').sort_values('date')
data_merge.set_index('date', inplace=True)
print('the num of NAN is: ', data_merge['outOctets'].isnull().sum(axis=0))
data_merge['outOctets'] = data_merge['outOctets'].interpolate(method='linear')
data_merge['outOctets'] = data_merge['outOctets'].rolling(window=5).mean()
data_merge.dropna(inplace=True)
diff_order = 4
data_shift = data_merge.shift(diff_order)
data_merge = data_merge.diff(diff_order).dropna()
print('Size:', data_merge.shape)

d = data_merge['outOctets'].describe()
delta = d.loc['75%'] - d.loc['25%']
low = d.loc['25%'] - 1.5 * delta
high = d.loc['75%'] + 1.5 * delta
print('high:', high)
print('low:', low)
data_merge.plot()
data_values = list(data_merge.iloc[:, 0].values.astype(np.float16))
print(len(data_values))
core_config = tf.ConfigProto()
# core_config.gpu_options.allow_growth = True
# gpu_num = os.getpid() % 2
# core_config.gpu_options.visible_device_list = str(gpu_num)
session = tf.Session(config=core_config)
keras.backend.set_session(session)


def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X, dtype=np.float16), np.array(y, dtype=np.float16)


def scheduler(epoch):
	if epoch % 10 == 0 and epoch != 0:
		lr = K.get_value(model.optimizer.lr)
		K.set_value(model.optimizer.lr, lr * 0.1)
		print("lr changed to {}".format(lr * 0.1))
	return K.get_value(model.optimizer.lr)


n_steps_in = 720
n_steps_out = 1
n_features = 1

X, y = split_sequence(data_values, n_steps_in, n_steps_out)
X = X.reshape((X.shape[0], X.shape[1], n_features))

test_num = 30
data_shift_tail = data_shift.tail(test_num).iloc[:, 0].values.tolist()
print(data_shift_tail)
X_train = X[:X.shape[0] - test_num, :, :]
print("X_train's shape is ", X_train.shape)
y_train = y[:X.shape[0] - test_num, :]
X_test = X[-test_num:, :, :]
y_test = y[-test_num:, :]
print('the shape of X_test is: ', X_test.shape)
print('the shape of y_test is: ', y_test.shape)
# define model
model = Sequential()
model.add(CuDNNLSTM(128, return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(CuDNNLSTM(128))
# model.add(LSTM(200, return_sequences=True, activation='sigmoid', input_shape=(n_steps_in, n_features)))
# model.add(LSTM(200, activation='sigmoid'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
# reduce_lr = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=2, verbose=2)
fit_start = time.time()
model.fit(X_train, y_train, epochs=500, batch_size=128, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
model.save("model_for_ckdk.hdf5")
fit_end = time.time()
print('Elapsed time of train is :', fit_end - fit_start)
yhat = model.predict(X_test, verbose=0)
print('y_hat:', type(yhat))
print(yhat.shape)
print('y_test:', type(y_test))
y_true_all = [i for item in y_test for i in item.tolist()]
y_true_all = list(map(lambda x: x[0] + x[1], zip(data_shift_tail, y_true_all)))
y_pred_all = [i for item in yhat for i in list(item)]
y_pred_all = list(map(lambda x: x[0] + x[1], zip(data_shift_tail, y_pred_all)))
y_pred_high = [i + high for i in y_pred_all]
y_pred_high = list(map(lambda x: x[0] + x[1], zip(data_shift_tail, y_pred_high)))
y_pred_low = [i + low for i in y_pred_all]
y_pred_low = list(map(lambda x: x[0] + x[1], zip(data_shift_tail, y_pred_low)))
assert len(y_pred_all) == len(y_true_all)
mae = mean_absolute_error(y_true_all, y_pred_all)
r2 = r2_score(y_true_all, y_pred_all)
mape = np.sum(list(map(lambda x, y: abs((x - y) / x) if x != 0 else abs(x - y) / 1, y_true_all, y_pred_all))) / len(
	y_pred_all)
print('error_mae_mean: ', mae)
print('error_mape_mean: ', mape)
print('r2_score is: ', r2)
print('y_true_all:', y_true_all)
print('y_pred_all:', y_pred_all)
print('y_pred_high:', y_pred_high)
print('y_pred_low:', y_pred_low)
plt.plot(y_true_all, label='true')
plt.plot(y_pred_all, label='pred')
plt.legend()
plt.show()

with open('ckdk.txt', 'a+') as f:
	y_true_str = ','.join([str(ele) for ele in y_true_all])
	y_pred_str = ','.join([str(ele) for ele in y_pred_all])
	f.write('y_true is: [' + y_true_str + ']\n')
	f.write('y_pred is: [' + y_pred_str + ']\n')
file_name = 'res_ckdk.pkl'
if os.path.exists(file_name):
	os.remove(file_name)
with open(file_name, 'wb') as f:
	pickle.dump(y_true_all, f)
	pickle.dump(y_pred_all, f)
	pickle.dump(y_pred_high, f)
	pickle.dump(y_pred_low, f)
end = time.time()
print('the total time is: ', end - begin)

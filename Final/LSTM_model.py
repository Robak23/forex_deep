
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
import os
import sys


nb_dir = os.getcwd()
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


# In[2]:


data = pd.read_csv("data/EURUSD_daily.csv", index_col='Date')
data.index = pd.to_datetime(data.index)
data.columns = ['close', 'open', 'high', 'low', 'pct']
data.drop('pct', axis=1, inplace=True)
data.sort_index(inplace=True)


# In[3]:


# add log returns and moving average
data['ret']  = np.log(data.close) - np.log(data.close.shift(1))
data['ma5']  = data.ret.rolling(5).mean()
data['ma20'] = data.ret.rolling(20).mean()


# In[4]:


# remove unstationarity from data
data.close = data.close - data.close.shift(1)
data.open = data.open - data.open.shift(1)
data.high = data.high - data.high.shift(1)
data.low = data.low - data.low.shift(1)


# In[5]:


data.dropna(inplace=True)


# In[6]:


for col in data.columns:
    data[col] = f.normalize(data[col])


# In[7]:


split = pd.Timestamp('01-01-2015')


# In[8]:


train = data.loc[:split,]
test = data.loc[split:,]


# In[9]:


for col in data.columns:
    train.loc[:,col], test.loc[:,col] = f.scale(train.loc[:,col], test.loc[:,col])


# In[34]:


x_train = train[:-1]
y_train = train.ma5.shift(-1)
y_train.dropna(inplace=True)

x_test = test[:-1]
y_test = test.ma5.shift(-1)
y_test.dropna(inplace=True)


# In[35]:


y_test


# In[36]:


x_test


# ### Training model

# In[37]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping


# In[38]:


batch_size = 32
epochs = 1000
validation_split = 0.05


# In[39]:


x_train_np = x_train.values
y_train_np = y_train.values

x_test_np = x_test.values
y_test_np = y_test.values


# In[40]:


x_train_t = x_train_np.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test_t = x_test_np.reshape(x_test.shape[0], 1, x_test.shape[1])


# In[41]:


early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)


# In[42]:


K.clear_session()

model = Sequential()

model.add(LSTM(100, input_shape= (x_train_t.shape[1], x_train_t.shape[2]), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(250, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mae', optimizer='adam', metrics=['accuracy', 'mae'])


# In[43]:


model.summary()


# In[44]:


history = model.fit(
    x_train_t,
    y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1, 
    callbacks=[early_stop], 
    validation_split=validation_split)


# In[45]:


eval_df = x_test


# In[46]:


eval_df['pred'] = model.predict(x_test_t, batch_size=batch_size)


# In[47]:


eval_df['real'] = y_test


# In[48]:


eval_df.head()


# In[50]:


eval_df.loc[:,['real','pred']].plot(figsize=(16,9))

plt.show()


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# In[2]:


data = pd.read_csv("../data/EURUSD_daily.csv", index_col='Date')


# In[3]:


data.index = pd.to_datetime(data.index)
data.columns = ['close']


# In[4]:


data.index.min()


# In[5]:


split_date = pd.Timestamp('01-01-2015')


# In[6]:


data['log_ret'] = np.log(data.close) - np.log(data.close.shift(1))


# In[7]:


data['pct_change'] = data.close.pct_change()


# In[8]:


data.head(5)


# In[9]:


mean = data.log_ret.mean()
std = data.log_ret.std()


# In[10]:


data['normalized'] = 1/(1+np.exp(-(data.log_ret-mean)/std))


# In[11]:


data['5MA'] = data.normalized.rolling(5).mean()


# In[12]:


data.dropna(inplace=True)


# In[13]:


data_n = data.drop('close', axis=1).drop('log_ret', axis=1).drop('pct_change', axis=1)


# In[14]:


train = data_n[:split_date]
test = data_n[split_date:]


# In[15]:


x_train = train[:-1]
y_train = train['5MA'][1:]

x_test = test[:-1]
y_test = test[1:]


# In[16]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras.backend as K
from keras.callbacks import EarlyStopping


# In[17]:


x_train_np = x_train.values
y_train_np = y_train.values

x_test_np = x_test.values
y_test_np = y_test.values


# In[18]:


x_train_t = x_train_np.reshape(x_train.shape[0], 1, 2)
x_test_t = x_test_np.reshape(x_test.shape[0], 1, 2)


# In[19]:


early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)


# In[20]:


K.clear_session()

model = Sequential()

model.add(LSTM(100, input_shape= (x_train_t.shape[1], x_train_t.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(150, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mse', optimizer='adam')


# In[21]:


model.summary()


# In[22]:


history = model.fit(x_train_t, y_train, epochs = 1000, batch_size=32, verbose = 1, callbacks=[early_stop])


# In[24]:


y_pred = model.predict(x_test_t, batch_size=32)


# In[28]:


fig = plt.figure(figsize = (16,9))
plt.plot(y_pred)
plt.plot(y_test_np)
plt.legend(['predicted', 'real'])


# In[29]:


model.evaluate(x=x_test_t, y=y_test_np, batch_size=32)


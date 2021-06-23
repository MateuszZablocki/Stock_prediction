import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

company = 'AMZN'  #używamy tu ticker symbol dla nazw firm

start = dt.datetime(2014,1,1)
end = dt.datetime(2021,6,1)

data = web.DataReader(company, 'yahoo', start, end) #pobiera dane z YahooDailyReader

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled)):
    x_train.append(scaled[x-prediction_days:x, 0])
    y_train.append(scaled[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Tworzenie modelu
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs = 25, batch_size = 32)

test_start = dt.datetime(2021,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(company, 'yahoo', test_start, test_end)
actual_price = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Tworzenie predykcji na testowych danych
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predict_price = model.predict(x_test)
predict_price = scaler.inverse_transform(predict_price)

#Tworzenie wykresu predykcji
plt.plot(actual_price, color="green", label=f"Obecna wycena {company}")
plt.plot(predict_price, color="red", label=f"Przewidziana wycena {company}")
plt.title(f"{company} Cena akcji")
plt.xlabel('Czas')
plt.ylabel(f"{company} Cena akcji")
plt.legend()
plt.show()

real_price = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
real_price = np.array(real_price)
real_price = np.reshape(real_price, (real_price.shape[0], real_price.shape[1], 1))

predicted_price = model.predict(real_price)
predicted_price = scaler.inverse_transform(predicted_price)
print(f"Przewidziana cena akcji kolejnego dnia dla {company}: {predicted_price}")








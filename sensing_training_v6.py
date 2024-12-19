import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Para
TRAIN_TRACE_FOLDER = './data_itsn/training_itsn_traces/training_sensing/'
PAST_STEPS = 400
FUTURE_STEPS = 4
EPOCHS = 60
BATCH_SIZE = 64
MODEL_FOLDER = "./module/sensing_module_v6.h5"
checkpoint_filepath = MODEL_FOLDER.replace('.h5', '_{epoch:02d}.h5')
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    save_freq='epoch',
    period=5,
    verbose=1,
    save_best_only=False
)

def load_bw_kbps(cooked_trace_folder):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_bw.append(float(parse[0]) / 1000.0)      # kbps
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_bw, all_file_names


def create_sequences(data, past_steps, future_steps):
    X, y = [], []
    for i in range(len(data) - past_steps - future_steps*100 + 1):
        end_index = i + past_steps
        seq_x = data[i: end_index]
        X.append(seq_x)
        for j in range(future_steps):
            seq_y = data[end_index+j*100: end_index+(j+1)*100]      # 100 is 1s
            y.append(np.mean(seq_y))

    return np.array(X), np.array(y)


def build_lstm_model(input_shape, future_step):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(input_shape, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dense(units=future_step, activation='relu'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


train_data, _ = load_bw_kbps(TRAIN_TRACE_FOLDER)
concatenated_train_data = np.vstack(train_data)
min_data = np.min(concatenated_train_data)
max_data = np.max(concatenated_train_data)
print('min training data, max training data: ', min_data, max_data)
train_scaler_data = (concatenated_train_data - min_data) / (max_data - min_data)


x_training, y_training = [], []
for train_data_i in train_scaler_data:
    x, y = create_sequences(train_data_i, PAST_STEPS, FUTURE_STEPS)
    x_training.append(x)
    y_training.append(y)

x_training = np.reshape(x_training, (-1, PAST_STEPS, 1))
y_training = np.reshape(y_training, (-1, FUTURE_STEPS, 1))

model = build_lstm_model(PAST_STEPS, FUTURE_STEPS)
model.fit(x_training, y_training, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
          callbacks=[model_checkpoint_callback])
model.save(MODEL_FOLDER)
print('========TRAINING DONE========')

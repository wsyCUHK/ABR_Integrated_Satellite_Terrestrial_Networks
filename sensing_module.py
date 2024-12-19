import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def sensing(model, past_ground):
    data = np.concatenate(past_ground)
    train_min, train_max = 0.0, 938025     # from training data
    past_steps, future_steps = 400, 4

    if len(data) < past_steps:
        data_filtered = np.array([value for value in data if value != 0])
        if len(data_filtered) > 0:
            h_mean = len(data_filtered) / np.sum(1.0 / data_filtered)
            interp_data = np.concatenate((data, np.full(past_steps - len(data), h_mean)))
        else:
            h_mean = 0
            interp_data = np.concatenate((data, np.full(past_steps - len(data), h_mean)))
    else:
        interp_data = data[-past_steps:]

    scaled_data = (interp_data - train_min) / (train_max - train_min)   # data is kbps
    input_data = np.reshape(scaled_data, (1, past_steps, 1))

    bw_predict_ = model.predict(input_data)     # shape is 4
    future_bw_ = ((bw_predict_ * (train_max - train_min)) + train_min)
    if np.std(interp_data) > 2000:
        future_bw_ *= 0.7
    else:
        future_bw_ *= 0.8

    bw_filtered = future_bw_[future_bw_ != 0]
    if len(bw_filtered) > 0:
        future_bw_h_mean_ = len(bw_filtered) / np.sum(1.0 / bw_filtered)
    else:
        future_bw_h_mean_ = 0

    future_bw = np.repeat(future_bw_, 100)

    future_bw = np.concatenate((future_bw, np.full(20000 - len(future_bw), future_bw_h_mean_)))

    return future_bw.reshape(-1,), np.std(future_bw_)

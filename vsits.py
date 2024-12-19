import copy
import cvxpy as cp
import numpy as np
from env import EnvITSNs
from utils import get_video_size


BITRATE_KBPS = np.array([20000, 40000, 60000, 80000, 110000, 160000])    # kbps
BITRATE_BPS = BITRATE_KBPS * 1e3  # bps
ACTION = np.append(BITRATE_KBPS, BITRATE_KBPS)    # kbps
LRNGTH = 4.0   # s
UE = 0.03       # delay of terrestrial network, ms
US = 0.05       # delay of satellite network, ms
VELOCITY_LIGHT = 3.0 * 1e8   # velocity of light
BITRATE_LEVELS = len(BITRATE_KBPS)
REBUFF_PENALTY_LINEAR = 160.0
REBUFF_PENALTY_LOG = 2.08
RANDOM_SEED = 42
TOTAL_VIDEO_CHUNK = 40
VIDEO_SIZE_FILE = './data_itsn/video/driving_4s_video_size_'
VIDEO_SIZE = get_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)  # in bytes


def opt_r_x(download_time, buffer, last_bitrate, last_rebuffer, qoe):
    optimal_chunk_number = len(download_time)
    bitrate_linear_value = np.concatenate([BITRATE_KBPS, BITRATE_KBPS]) / 1000.0
    bitrate_log_value = np.log(bitrate_linear_value / np.min(bitrate_linear_value))
    R = bitrate_linear_value.size

    r = cp.Variable((optimal_chunk_number, R), boolean=True)
    b = cp.Variable(optimal_chunk_number, nonneg=True)
    z = cp.Variable(optimal_chunk_number, nonneg=True)

    ue = cp.Parameter(pos=True)
    ue.value = UE
    us = cp.Parameter(pos=True)
    us.value = US
    L = cp.Parameter(pos=True)
    L.value = LRNGTH

    T = cp.Parameter((optimal_chunk_number, R), pos=True)
    T.value = download_time
    bitrate_linear = cp.Parameter(shape=R, pos=True)
    bitrate_linear.value = bitrate_linear_value
    bitrate_log = cp.Parameter(shape=R, pos=True)
    bitrate_log.value = bitrate_log_value

    weight = cp.Parameter((optimal_chunk_number,), pos=True)
    weight.value = np.array([0.00001 for _ in range(optimal_chunk_number)])
    tool1 = cp.Parameter(R, boolean=True)
    tool2 = cp.Parameter(R, boolean=True)

    a = np.zeros((R,))
    for k in range(int(R / 2)):
        a[k] = 1
    tool1.value = a
    a_ = np.ones((R,))
    tool2.value = a_

    e1 = np.ones((R, 1))
    e2 = np.ones((optimal_chunk_number, 1))
    e3 = np.zeros((1, optimal_chunk_number))
    e3[0][0] = 1
    if last_bitrate < R / 2.0 - 0.5:
        last_path = 1  # ground
    else:
        last_path = 0  # leo

    cons = [r @ e1 == e2]

    for t in range(optimal_chunk_number):
        cons += [b[t] <= 60.0]
        cons += [b[t] >= 4.0]
    cons += [b[0] == buffer]
    if optimal_chunk_number != 1:
        cons += [b[0] - (cp.multiply(r, T) @ tool2)[0]
                 - ((r @ tool1)[0] - last_path) * (us - ue) / 2
                 - cp.abs((r @ tool1)[0] - last_path) * (ue + us) / 2.0
                 + z[0] + L >= b[1]]
    for t in range(1, optimal_chunk_number - 1):
        cons += [b[t] - (cp.multiply(r, T) @ tool2)[t]
                 - ((r @ tool1)[t] - (r @ tool1)[t - 1]) * (us - ue) / 2
                 - cp.tv((r @ tool1)[t - 1:t + 1]) * (ue + us) / 2.0
                 + z[t] + L >= b[t + 1]]

    cons += [z[0] >= (cp.multiply(r, T) @ tool2)[0]
             + ((r @ tool1)[0] - last_path) * (us - ue) / 2
             + cp.abs((r @ tool1)[0] - last_path) * (ue + us) / 2.0 - b[0]]     ###################################
    for t in range(1, optimal_chunk_number):
        cons += [z[t] >= (cp.multiply(r, T) @ tool2)[t]
                 + ((r @ tool1)[t] - (r @ tool1)[t - 1]) * (us - ue) / 2
                 + cp.tv((r @ tool1)[t - 1:t + 1]) * (ue + us) / 2.0 - b[t]]

    if qoe == 'linear':
        obj = cp.Minimize(cp.sum(-1 * r @ bitrate_linear + REBUFF_PENALTY_LINEAR * z)
                          + cp.tv(r @ bitrate_linear)
                          - bitrate_linear_value[last_bitrate] + REBUFF_PENALTY_LINEAR * last_rebuffer
                          + cp.abs(bitrate_linear_value[last_bitrate] - (r @ bitrate_linear)[0]))
    elif qoe == 'log':
        obj = cp.Minimize(cp.sum(-1 * r @ bitrate_log + REBUFF_PENALTY_LOG * z) + cp.tv(r @ bitrate_log)
                          - bitrate_log_value[last_bitrate] + REBUFF_PENALTY_LOG * last_rebuffer
                          + cp.abs(bitrate_log_value[last_bitrate] - (r @ bitrate_log)[0]))

    prob = cp.Problem(obj, cons)

    # print("prob is DQCP:", prob.is_dqcp())
    # print("obj is DQCP:", obj.is_dqcp())
    # print("cons[0] is DQCP:", cons[0].is_dqcp())
    # print("cons[1] is DQCP:", cons[1].is_dqcp())
    # print("cons[2] is DQCP:", cons[2].is_dqcp())
    # print("cons[3] is DQCP:", cons[3].is_dqcp())
    # print("cons[-1] is DQCP:", cons[-1].is_dqcp())

    prob.solve(verbose=False, solver='GUROBI')

    return r.value, b.value, prob.value, z.value


def opt_c(t_j, future_chunk_number, leo_trans_rate, ground_trans_rate, leo_user_distance):
    clock_now = 0
    c_s = np.zeros((future_chunk_number,))
    d_s = np.zeros((future_chunk_number,))
    for i, j in enumerate(t_j):
        c_s[i] = np.sum(leo_trans_rate[clock_now: int(j * 100 + clock_now)] * 0.01) / j
        d_s[i] = np.max(leo_user_distance[clock_now: int(j * 100 + clock_now)])
        clock_now += int(j * 100.0)
        clock_now = round(clock_now, 2)

    clock_now = 0
    c_e = np.zeros((future_chunk_number,))
    for i, j in enumerate(t_j):
        c_e[i] = np.sum(ground_trans_rate[clock_now: int(j * 100 + clock_now)] * 0.01) / j
        clock_now += int(j * 100.0)
        clock_now = round(clock_now, 2)
    return c_s, d_s, c_e


def init_download_time(c_s, c_e, d_s, future_chunk_number, chunk_index, past_ground_std):
    # leo trans rate, c_s
    # ground trans rate, c_e
    # leo_user_distance, d_s
    if chunk_index + future_chunk_number <= TOTAL_VIDEO_CHUNK:
        t = np.zeros((future_chunk_number, len(ACTION)))
        for i, bit in enumerate(BITRATE_BPS):
            for j in range(future_chunk_number): 
                leo_bw = max(np.mean(c_s[j*400:(j+1)*400]), 0.01)
                t[j][i] = (VIDEO_SIZE[i][j + chunk_index] * 8.0 * 0.001 / leo_bw) \
                          + (np.max(d_s[j*400:(j+1)*400]) / VELOCITY_LIGHT)

                ground_bw = max(np.mean(c_e[j*200:(j+1)*200])
                                - (j/future_chunk_number + 1.2) * past_ground_std, 0.01)
                t[j][i + len(BITRATE_BPS)] = VIDEO_SIZE[i][j + chunk_index] \
                                             * 8.0 * 0.001 / ground_bw

    else:
        t = np.zeros((TOTAL_VIDEO_CHUNK - chunk_index, len(ACTION)))
        for i, bit in enumerate(BITRATE_BPS):
            for j in range(TOTAL_VIDEO_CHUNK - chunk_index):
                leo_bw = max(np.mean(c_s[j*400:(j+1)*400]), 0.01)
                t[j][i] = (VIDEO_SIZE[i][j + chunk_index] * 8.0 * 0.001 / leo_bw) \
                          + (np.max(d_s[j*400:(j+1)*400]) / VELOCITY_LIGHT)

                ground_bw = max(np.mean(c_e[j*200:(j+1)*200])
                                - (j/future_chunk_number + 1.2) * past_ground_std, 0.01)
                t[j][i + len(BITRATE_BPS)] = VIDEO_SIZE[i][j + chunk_index] \
                                             * 8.0 * 0.001 / ground_bw

    return t


def update_download_time(c_s_new, c_e_new, d_s_new, future_chunk_number, chunk_index):
    if chunk_index + future_chunk_number <= TOTAL_VIDEO_CHUNK:
        t = np.zeros((future_chunk_number, len(ACTION)))
        for i, bit in enumerate(BITRATE_BPS):
            for j in range(future_chunk_number):
                if c_e_new[j] == 0.0:
                    c_e_new[j] += 0.001
                t[j][i] = (VIDEO_SIZE[i][j+chunk_index] * 8.0 * 0.001 / c_s_new[j]) +\
                          (d_s_new[j] / VELOCITY_LIGHT)
                t[j][i + len(BITRATE_BPS)] = VIDEO_SIZE[i][j+chunk_index] * 8.0 * 0.001 / c_e_new[j]
    else:
        t = np.zeros((TOTAL_VIDEO_CHUNK - chunk_index, len(ACTION)))
        for i, bit in enumerate(BITRATE_BPS):
            for j in range(TOTAL_VIDEO_CHUNK - chunk_index):
                if c_e_new[j] == 0.0:
                    c_e_new[j] += 0.001
                t[j][i] = (VIDEO_SIZE[i][j+chunk_index] * 8.0 * 0.001 / c_s_new[j]) + (d_s_new[j] / VELOCITY_LIGHT)
                t[j][i + len(BITRATE_BPS)] = VIDEO_SIZE[i][j+chunk_index] * 8.0 * 0.001 / c_e_new[j]

    return t


def get_chunk_bitrate(leo_trans_rate, ground_trans_rate, leo_user_distance,
                      future_chunk_number, buffer, chunk_index,
                      last_chunk_bitrate, last_rebuffer, qoe, past_ground_std):

    # init env
    downloadT_init = init_download_time(leo_trans_rate, ground_trans_rate,
                                        leo_user_distance, future_chunk_number,
                                        chunk_index, past_ground_std)

    optimal_bitrate, optimal_buffer, \
    optimal_qoe, optimal_rebuffer = opt_r_x(downloadT_init, buffer,
                                            last_chunk_bitrate, last_rebuffer, qoe)

    c_s_old = 0
    c_e_old = 0
    qoe_list = []
    while True:
        env_itsn_opt = EnvITSNs(leo_trans_rate, ground_trans_rate, leo_user_distance,
                                random_seed=RANDOM_SEED, opt=True)

        T_j = []
        D_j = []
        for k, i in enumerate(optimal_bitrate):
            index = np.argwhere(i == 1)[0][0]
            T, _, _, _, _, _, _, _, _, d_j, _, _, _, _ = env_itsn_opt.update_env(index, k+chunk_index, 'none')
            T_j.append(T)
        c_s, d_s, c_e = opt_c(T_j, future_chunk_number,
                              leo_trans_rate, ground_trans_rate, leo_user_distance)

        if np.sum(c_s - c_s_old) <= 1 and np.sum(c_e - c_e_old) <= 1:
            downloadT_update = update_download_time(c_s, c_e, d_s, future_chunk_number, chunk_index)
            optimal_bitrate, optimal_buffer, \
                optimal_qoe, optimal_rebuffer = opt_r_x(downloadT_update, buffer,
                                                        last_chunk_bitrate, last_rebuffer, qoe)
            qoe_list.append(np.round(optimal_qoe, 4))

            return np.argwhere(optimal_bitrate[0] == 1)[0][0]
        else:
            c_s_old = copy.deepcopy(c_s)
            c_e_old = copy.deepcopy(c_e)

        downloadT_update = update_download_time(c_s, c_e, d_s, future_chunk_number, chunk_index)
        if np.any(downloadT_update <= 0):
            print(downloadT_update)
        optimal_bitrate, optimal_buffer, \
            optimal_qoe, optimal_rebuffer = opt_r_x(downloadT_update, buffer,
                                                    last_chunk_bitrate, last_rebuffer, qoe)
        qoe_list.append(np.round(optimal_qoe, 4))
        
        if len(qoe_list) > 10:
            if np.round(optimal_qoe, 4) == min(qoe_list[-8:]):
                return np.argwhere(optimal_bitrate[0] == 1)[0][0]
            if np.round(optimal_qoe, 4) == qoe_list[-2] == qoe_list[-3] == qoe_list[-4] == qoe_list[-5]:
                return np.argwhere(optimal_bitrate[0] == 1)[0][0]

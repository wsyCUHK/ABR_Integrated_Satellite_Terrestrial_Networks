import numpy as np
from copy import deepcopy
from utils import get_video_size
import os
from env import EnvITSNs
from vsits import get_chunk_bitrate
from sensing_module import sensing
import tensorflow as tf
import argparse
from time import time


parser = argparse.ArgumentParser(description='ITSN ABR Ours')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument("--sensing", type=str, default='v6', help="sensing module")  # 'none' is ours
parser.add_argument("--qoe", type=str, default='log', help="QoE types")
parser.add_argument("--future", type=int, default=4, help="future_chunk_number")
args = parser.parse_args()


FUTURE_CHUNK_NUMBER = args.future
if args.sensing != 'none':
    SENSING = './module/sensing_module_' + args.sensing + '_35.h5'
    model = tf.keras.models.load_model(SENSING)


BITRATE_KBPS = np.array([20000, 40000, 60000, 80000, 110000, 160000])    # kbps
BITRATE_BPS = BITRATE_KBPS * 1e3  # bps
ACTION = np.concatenate([BITRATE_KBPS, BITRATE_KBPS])    # kbps
TOTAL_VIDEO_CHUNK = 40
M_IN_K = 1000.0
L = 4   # s
UE = 0.03       # delay of terrestrial network, ms
US = 0.05       # delay of satellite network, ms
REBUFF_PENALTY_LINEAR = 160.0
REBUFF_PENALTY_LOG = 2.08
RANDOM_SEED = 42
BITRATE_LEVELS = len(BITRATE_KBPS)
DEFAULT_QUALITY = len(BITRATE_KBPS)  # default video quality without agent

VIDEO_SIZE_FILE = './data_itsn/video/driving_4s_video_size_'
TEST_DATA_FOLDER = './data_itsn/testing_itsn_traces/'
LEO_FOLDER = 'testing_leo_traces/'
GROUND_FOLDER = 'testing_ground_traces/'
LEO_DIST_FOLDER = 'testing_leo_user_dist/'

QoE = args.qoe
TEST_LOG_FOLDER = './ours_test_results_' + QoE + '_' \
                  + str(args.future) + '_' + str(args.seed) + '/'
LOG_FILE = TEST_LOG_FOLDER + 'log_sim_ours_'
if not os.path.exists(TEST_LOG_FOLDER):
    os.makedirs(TEST_LOG_FOLDER)

leo_files = os.listdir(TEST_DATA_FOLDER + LEO_FOLDER)
video_size = get_video_size(VIDEO_SIZE_FILE, BITRATE_LEVELS)  # in bytes

decision_making_time = []

print('Begin')

np.random.seed(args.seed)

for leo_file in leo_files:
    trace_idx = leo_file.split('_')[-1]
    leo_trans_rate_ = np.loadtxt(TEST_DATA_FOLDER + LEO_FOLDER + 'leo_trace_001interval_' + trace_idx) # bps
    ground_trans_rate_ = np.loadtxt(TEST_DATA_FOLDER + GROUND_FOLDER + 'ground_trace_001interval_' + trace_idx)
    leo_user_distance_ = np.loadtxt(TEST_DATA_FOLDER + LEO_DIST_FOLDER + 'leo_user_dist_001interval_' + trace_idx)

    leo_trans_rate = np.concatenate([leo_trans_rate_, leo_trans_rate_, leo_trans_rate_]) / 1000.0      # kbps
    ground_trans_rate = np.concatenate([ground_trans_rate_, ground_trans_rate_, ground_trans_rate_]) / 1000.0      # kbps
    leo_user_distance = np.concatenate([leo_user_distance_, leo_user_distance_, leo_user_distance_])     # m

    # if trace_idx != '7':
    #     continue
    print('trace id: ', trace_idx)
    # init env
    env_itsn_test = EnvITSNs(leo_trans_rate, ground_trans_rate, leo_user_distance, random_seed=RANDOM_SEED)

    log_path = LOG_FILE + str(trace_idx)
    log_file = open(log_path, 'w')

    last_bit_rate = DEFAULT_QUALITY     # 6, default is ground bitrate 0
    bit_rate = DEFAULT_QUALITY

    time_stamp = 0
    next_chunk_index = 0
    past_ground_ = []

    while True:
        total_time, sleep_time, buffer_size, rebuffer, video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, path, _, future_leo_trans_rate, \
        future_leo_user_dist, past_ground_trans_rate, \
        next_chunk_index = env_itsn_test.update_env(bit_rate, next_chunk_index, args.sensing)
        # print('next_chunk_index ', next_chunk_index)
        time_stamp += total_time  # in s
        time_stamp += sleep_time  # in s

        if QoE == 'linear':
            reward = ACTION[bit_rate] / M_IN_K \
                     - REBUFF_PENALTY_LINEAR * rebuffer \
                     - np.abs(ACTION[bit_rate] - ACTION[last_bit_rate]) / M_IN_K

        elif QoE == 'log':
            log_bit_rate = np.log(ACTION[bit_rate] / float(ACTION[0]))
            log_last_bit_rate = np.log(ACTION[last_bit_rate] / float(ACTION[0]))
            reward = log_bit_rate - REBUFF_PENALTY_LOG * rebuffer - np.abs(log_bit_rate - log_last_bit_rate)

        else:
            assert QoE == 'error'
            reward = '0'

        last_bit_rate = deepcopy(bit_rate)

        log_file.write(str(time_stamp) + '\t' +
                       str(ACTION[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuffer) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(total_time) + '\t' +
                       str(path) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        if end_of_video:
            log_file.write('\n')
            log_file.close()
            break

        if args.sensing == 'none':
            future_ground_trans_rate = deepcopy(past_ground_trans_rate)
            past_ground_std = 0
        else:
            past_ground_.append(past_ground_trans_rate)
            future_ground_trans_rate, past_ground_std = sensing(model, past_ground_)

        start_time = time()
        bit_rate = get_chunk_bitrate(future_leo_trans_rate, future_ground_trans_rate,
                                     future_leo_user_dist, FUTURE_CHUNK_NUMBER, buffer_size,
                                     next_chunk_index, last_bit_rate, rebuffer, QoE, past_ground_std)
        decision_making_time.append(time() - start_time)
        np.savetxt(TEST_LOG_FOLDER+'ours_' + str(args.future) + 'N_decision_making_time.txt',
                   [np.mean(decision_making_time)], fmt='%.6f')



rewards, entropies = [], []
test_log_files = os.listdir(TEST_LOG_FOLDER)

for test_log_file in test_log_files:
    if 'log' not in test_log_file:
        continue
    reward = []
    with open(TEST_LOG_FOLDER + test_log_file, 'r') as f:

        for line in f:
            parse = line.split()
            try:
                reward.append(float(parse[-1]))
            except IndexError:
                break
    rewards.append(np.sum(reward[1:]))
    print(test_log_file, np.sum(reward[1:]))
rewards = np.array(rewards)
rewards_mean = np.mean(rewards)
print('files: ', len(test_log_files))
print('ours ITSNs reward: ', rewards_mean)



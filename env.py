import numpy as np
from copy import deepcopy


VIDEO_CHUNK_LEN = 4.0
BITRATE = np.array([20000, 40000, 60000, 80000, 110000, 160000])    # kbps
RANDOM_SEED = 42
BUFFER_THRESH = 60.0  # sec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 0.5  # sec
BS_LINK_DELAY = 0.03
LEO_LINK_DELAY = 0.05
V_LIGHT = 3.0 * 1e8
BITS_IN_BYTE = 8.0
TOTAL_CHUNK = 40
BITRATE_LEVELS = len(BITRATE)
ACTION = np.append(BITRATE, BITRATE)
VIDEO_SIZE_FILE = './data_itsn/video/driving_4s_video_size_'


class EnvITSNs:
    def __init__(self, all_leo_cooked_bw, all_bs_cooked_bw, all_leo_user_distance,
                 random_seed=RANDOM_SEED, opt=False):

        np.random.seed(random_seed)

        self.leo_trans_rate = all_leo_cooked_bw     # kbp s
        self.ground_trans_rate = all_bs_cooked_bw
        self.leo_user_distance = all_leo_user_distance  # m

        self.video_chunk_counter = 0
        if opt:
            self.buffer_size = 4.0
        else:
            self.buffer_size = 0.0

        self.clock = 0      # index
        self.path = 0       # x = 0, 1
        self.last_path = 0      # x = 0, 1

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def update_env(self, a_index, counter, sensing='v1'):
        current_chunk_bitrate = ACTION[a_index]
        index = np.argwhere(current_chunk_bitrate == BITRATE)[0][0]
        video_chunk_size = self.video_size[index][counter]     # byte/s

        if a_index < ACTION.size / 2.0 - 0.5:
            self.path = 1     # leo
        else:
            self.path = 0     # ground

        if self.path < 0.5:
            used_index = np.argwhere(np.cumsum(self.ground_trans_rate[self.clock:] * 0.01)
                                    >= video_chunk_size * BITS_IN_BYTE * 0.001)[0][0] + 1  # start from 0
            download_time = used_index * 0.01        # s
            propagation_delay = 0.0
            if self.last_path < 0.5:
                setup_delay = 0.0
            else:
                setup_delay = BS_LINK_DELAY
            return_leo_user_distance = self.leo_user_distance[self.clock: (self.clock + used_index)]

        else:
            used_index = np.argwhere(np.cumsum(self.leo_trans_rate[self.clock:] * 0.01)
                                    >= video_chunk_size * BITS_IN_BYTE * 0.001)[0][0] + 1
            download_time = used_index * 0.01        # s
            max_dist = np.max(self.leo_user_distance[self.clock: (self.clock + used_index)])
            propagation_delay = max_dist / V_LIGHT
            if self.path > 0.5:
                setup_delay = 0.0
            else:
                setup_delay = LEO_LINK_DELAY
            return_leo_user_distance = self.leo_user_distance[self.clock: (self.clock + used_index)]

        total_time = download_time + propagation_delay + setup_delay    # sec
        T = int(np.round(total_time * 100.0))    # index

        if sensing != 'none':
            past_ground_trans_rate = self.ground_trans_rate[self.clock: self.clock + T]

        self.clock = self.clock + T

        rebuffer = np.maximum(total_time - self.buffer_size, 0.0)

        self.buffer_size = np.maximum(self.buffer_size - total_time, 0) + VIDEO_CHUNK_LEN

        self.last_path = self.path
        return_path = deepcopy(self.path)

        sleep_time = 0      # sec
        if self.buffer_size > BUFFER_THRESH:
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time
            sleep_index = int(np.round(sleep_time * 100.0))  # index
            self.clock = self.clock + sleep_index  # index

        return_buffer_size = self.buffer_size
        self.video_chunk_counter = counter + 1
        next_chunk_index = self.video_chunk_counter
        video_chunk_remain = TOTAL_CHUNK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_CHUNK:
            end_of_video = True
            self.video_chunk_counter = 0
            self.buffer_size = 0.0

            self.clock = 0  # index
            self.path = 0  # x = 0, 1
            self.last_path = 0  # x = 0, 1

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        future_leo_trans_rate = self.leo_trans_rate[self.clock:]
        future_leo_user_dist = self.leo_user_distance[self.clock:]

        if sensing == 'none':
            past_ground_trans_rate = self.ground_trans_rate[self.clock:]

        return total_time, sleep_time, return_buffer_size, rebuffer, \
               video_chunk_size, next_video_chunk_sizes, end_of_video, \
               video_chunk_remain, return_path, return_leo_user_distance, \
               future_leo_trans_rate, future_leo_user_dist, past_ground_trans_rate, next_chunk_index

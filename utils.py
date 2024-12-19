import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_video_size(video_size_file, bitrate_levels):
    video_size = {}  # in bytes
    for bitrate in range(bitrate_levels):
        video_size[bitrate] = []
        with open(video_size_file + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
    return video_size


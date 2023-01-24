import json


import logging
import time
import os

def yield_train(in_file):
    with open(in_file, 'r') as f:
        train_json = json.load(f)

    for vid in train_json:
        record = train_json[vid]
        for t, s in zip(record["timestamps"], record["sentences"]):
            yield t, s

def time_str(ts):
    res = "{:.2f} "*len(ts) + "|"
    return res.format(*ts)


def get_logger(dir, tile):
    os.makedirs(dir, exist_ok=True)
    log_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(dir, "{}_{}.log".format(log_file, tile))

    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(levelname)s:%(message)s"
    # DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT)
    # chlr = logging.StreamHandler() # 输出到控制台的handler
    # chlr.setFormatter(formatter)
    # chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
    fhlr = logging.FileHandler(log_file) # 输出到文件的handler
    fhlr.setFormatter(formatter)
    # logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
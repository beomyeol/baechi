import argparse
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression

from tensorflow.python.client import timeline

from utils import logger

_LOGGER = logger.get_logger(__file__)


def run_benchmark(tensor_size, from_gpu_id, to_gpu_id,
                  warmup_count=5, num_measurements=20):
    with tf.Graph().as_default():
        with tf.device('/device:GPU:%d' % from_gpu_id):
            x1 = tf.get_variable("var1a", [tensor_size, 1])
            x2 = tf.get_variable("var1b", [1, 1])
            from_op = tf.matmul(x1, x2, name='from_op')
        with tf.device('/device:GPU:%d' % to_gpu_id):
            y = tf.get_variable("var2", [1, 1])
            out = tf.matmul(from_op, y, name='to_op')

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # warm up
            for _ in range(warmup_count):
                sess.run(out)

            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata_list = []
            for _ in range(num_measurements):
                run_metadata = tf.RunMetadata()
                sess.run(out, options=run_options, run_metadata=run_metadata)

                run_metadata_list.append(run_metadata)
            return run_metadata_list


def get_transfer_time(timeline_json, from_op_name='from_op',
                      to_op_name='to_op'):
    data = timeline_json['traceEvents']
    end_ts = start_ts = None
    for dic in data:
        for key, value in dic.items():
            if key == 'cat' and value == 'Op':
                for key, value in dic.items():
                    if key == "args" and value['name'] == from_op_name:
                        new_end_ts = dic['ts'] + dic['dur']
                        end_ts = max(end_ts or new_end_ts, new_end_ts)
                    if key == "args" and value['name'] == to_op_name:
                        new_start_ts = dic['ts']
                        start_ts = min(start_ts or new_start_ts, new_start_ts)
    transfer_time = start_ts - end_ts
    assert transfer_time > 0
    return transfer_time


def generate_dataset(results):
    transfer_times_by_size = []

    for tensor_size, run_metadata_list in results:
        transfer_times = []
        for run_metadata in run_metadata_list:
            chrome_trace_str = timeline.Timeline(
                run_metadata.step_stats).generate_chrome_trace_format()
            timeline_json = json.loads(chrome_trace_str)
            transfer_times.append(get_transfer_time(timeline_json))
        transfer_times_by_size.append((tensor_size, transfer_times))

    X = []
    Y = []
    for x, ys in transfer_times_by_size:
        for y in ys:
            X.append(x * 4)
            Y.append(y)

    return X, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_gpu_id', type=int, default=0,
                        help='From GPU ID')
    parser.add_argument('--to_gpu_id', type=int, default=1,
                        help='To GPU ID')
    parser.add_argument('--exponent', type=int, default=30,
                        help='Max tensor size. 2^(exponent).')

    args = parser.parse_args()

    tensor_sizes = [2 ** i for i in range(0, args.exponent)]

    _LOGGER.info('Running benchmark to measure communication costs')

    results = []
    for tensor_size in tensor_sizes:
        run_metadata_list = run_benchmark(
            tensor_size, args.from_gpu_id, args.to_gpu_id)
        results.append((tensor_size, run_metadata_list))

    X, Y = generate_dataset(results)

    reg = LinearRegression().fit([[x] for x in X], [[y] for y in Y])
    print('Communication cost function: {} x + {}'.format(
        reg.coef_[0][0], reg.intercept_[0]))


if __name__ == "__main__":
    main()

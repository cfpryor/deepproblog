# Lint as: python3
import json
import os
import shutil
import time

import addition

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(THIS_DIR, '..', '..', '..', '..', '..', 'psl_data')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
LOG_DIR = os.path.join(THIS_DIR, 'log')

EPOCHS = {1: {20: 30, 30: 30, 37: 30, 40: 30, 75: 40, 150: 40, 300: 20, 3000: 6, 25000: 10},
          2: {10: 30, 15: 30, 18: 30, 20: 30, 37: 30, 75: 40, 150: 40, 1500: 8, 12500: 10}}
METHOD = "exact"


def write_json(data, path):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_data(path):
    indices = []

    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()[:-1]
            for part in parts:
                indices.append(int(part))

    return indices


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for experiment_dir in sorted(os.listdir(DATA_DIR)):
        experiment_path = os.path.join(DATA_DIR, experiment_dir)
        experiment_result_path = os.path.join(RESULTS_DIR, experiment_dir)
        os.makedirs(experiment_result_path, exist_ok=True)
        num_digits = int(experiment_dir[-1:])

        for split_dir in sorted(os.listdir(experiment_path)):
            split_path = os.path.join(experiment_path, split_dir)
            if os.path.isfile(split_path):
                continue
            split_result_path = os.path.join(experiment_result_path, split_dir)
            os.makedirs(split_result_path, exist_ok=True)
            split = int(split_path[-2:])
            if split < 0:
                continue

            for train_size_dir in sorted(os.listdir(split_path)):
                train_size_path = os.path.join(split_path, train_size_dir)
                train_size_result_path = os.path.join(split_result_path, train_size_dir)
                os.makedirs(train_size_result_path, exist_ok=True)
                train_size = int(train_size_path[-4:])

                for overlap_dir in sorted(os.listdir(train_size_path)):
                    overlap_path = os.path.join(train_size_path, overlap_dir)
                    overlap_result_path = os.path.join(train_size_result_path,
                                                       overlap_dir)
                    os.makedirs(overlap_result_path, exist_ok=True)

                    imagesum_learn_path = os.path.join(overlap_path, 'image-sum-truth-train.txt')
                    imagesum_eval_path = os.path.join(overlap_path, 'image-sum-truth-test.txt')

                    config_path = os.path.join(overlap_result_path, "config.json")
                    log_path = os.path.join(overlap_result_path, "out.txt")
                    print("Starting Run: %s" % (overlap_path,))
                    if os.path.isfile(config_path):
                        print("Found existing config file, skipping run: %s" % (overlap_path,))
                        continue

                    train_indices = load_data(imagesum_learn_path)
                    val_indices = load_data(imagesum_eval_path)

                    time_start = time.time()
                    config = addition.main(method=METHOD,
                                           num_digits=num_digits,
                                           seed=split,
                                           num_epochs=EPOCHS[num_digits][train_size],
                                           batch_size=2,
                                           learning_rate=1e-3,
                                           log_iterations=train_size // 5,
                                           train_indices=train_indices,
                                           test_indices=val_indices,
                                           train_set_name="train",
                                           test_set_name="train")
                    time_end = time.time()

                    config["program_time_start"] = time_start
                    config["program_time_end"] = time_end
                    config["program_total_time"] = (time_end - time_start)

                    saved_log_path = os.path.join(LOG_DIR, config["name"] + ".log")

                    shutil.move(saved_log_path, log_path)
                    write_json(config, config_path)


if __name__ == '__main__':
    main()

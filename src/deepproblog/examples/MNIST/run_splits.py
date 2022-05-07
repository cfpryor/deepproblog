# Lint as: python3
import json
import os
import time

import addition

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(THIS_DIR, '..', '..', '..', '..', '..', 'psl_data',
                        'mnist-addition')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')
LOG_DIR = os.path.join(THIS_DIR, 'log')

EVAL_DIR = 'eval'
LEARN_DIR = 'learn'
IMAGESUM_FILENAME = 'imagesum_truth.txt'
CONFIG_FILENAME = 'config.json'
LOG_FILENAME = 'out.log'


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

        for fold_dir in sorted(os.listdir(experiment_path)):
            fold_path = os.path.join(experiment_path, fold_dir)
            if os.path.isfile(fold_path):
                continue
            fold_result_path = os.path.join(experiment_result_path, fold_dir)
            os.makedirs(fold_result_path, exist_ok=True)
            fold = int(fold_path[-2:])

            for train_size_dir in sorted(os.listdir(fold_path)):
                train_size_path = os.path.join(fold_path, train_size_dir)
                train_size_result_path = os.path.join(fold_result_path,
                                                      train_size_dir)
                os.makedirs(train_size_result_path, exist_ok=True)
                train_size = int(train_size_path[-5:])

                for overlap_dir in sorted(os.listdir(train_size_path)):
                    overlap_path = os.path.join(train_size_path, overlap_dir)
                    overlap_result_path = os.path.join(train_size_result_path,
                                                       overlap_dir)
                    os.makedirs(overlap_result_path, exist_ok=True)

                    imagesum_learn_path = os.path.join(overlap_path, LEARN_DIR,
                                                       IMAGESUM_FILENAME)
                    imagesum_eval_path = os.path.join(overlap_path, EVAL_DIR,
                                                      IMAGESUM_FILENAME)

                    config_path = os.path.join(overlap_result_path,
                                               CONFIG_FILENAME)
                    log_path = os.path.join(overlap_result_path, LOG_FILENAME)
                    print("Starting Run: %s" % (overlap_path,))
                    if os.path.isfile(config_path):
                        print("Found existing config file, skipping run: %s" % (
                            overlap_path,))
                        continue

                    train_indices = load_data(imagesum_learn_path)
                    test_indices = load_data(imagesum_eval_path)

                    time_start = time.time()
                    if train_size >= 1000:
                        num_epochs = 5
                    else:
                        num_epochs = 15
                    config = addition.main(num_digits=num_digits,
                                           seed=fold,
                                           num_epochs=num_epochs,
                                           batch_size=2,
                                           learning_rate=1e-3,
                                           log_iterations=train_size // 5,
                                           train_indices=train_indices,
                                           test_indices=test_indices,
                                           train_set_name="train",
                                           test_set_name="train")
                    time_end = time.time()
                    config["program_time_start"] = time_start
                    config["program_time_end"] = time_end
                    config["program_total_time"] = (time_end - time_start)

                    saved_log_path = os.path.join(LOG_DIR, config["name"] + ".log")

                    os.rename(saved_log_path, log_path)
                    write_json(config, config_path)


if __name__ == '__main__':
    main()

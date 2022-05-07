# Lint as: python3
import csv
import json
import os

import numpy

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

EVAL_DIR = 'eval'
LEARN_DIR = 'learn'
IMAGESUM_FILENAME = 'imagesum_truth.txt'
CONFIG_FILENAME = 'config.json'
LOG_FILENAME = 'out.log'
CSV_RESULTS = 'results.csv'


def load_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)

def write_csv(data, path):
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(data)

def print_results(results):
    for name in results:
        mean = numpy.mean(results[name])
        stdev = numpy.std(results[name])
        print("Experiment: %s  Average Accuracy: %0.4f  Stdev: %0.4f" % (name, mean, stdev))


def main():
    results = {}
    csv_results = [["Experiment", "Fold", "Train Size", "Overlap", "Accuracy"]]
    for experiment_dir in sorted(os.listdir(RESULTS_DIR)):
        experiment_result_path = os.path.join(RESULTS_DIR, experiment_dir)
        for fold_dir in sorted(os.listdir(experiment_result_path)):
            fold_result_path = os.path.join(experiment_result_path, fold_dir)
            for train_size_dir in sorted(os.listdir(fold_result_path)):
                train_size_result_path = os.path.join(fold_result_path,
                                                      train_size_dir)
                for overlap_dir in sorted(os.listdir(train_size_result_path)):
                    overlap_result_path = os.path.join(train_size_result_path,
                                                       overlap_dir)
                    config_result_path = os.path.join(overlap_result_path,
                                                      CONFIG_FILENAME)
                    if not os.path.isfile(config_result_path):
                        continue

                    config = load_json(config_result_path)
                    experiment_name = experiment_dir + "-" + train_size_dir + "-" + overlap_dir

                    if experiment_name not in results:
                        results[experiment_name] = []

                    results[experiment_name].append(config["accuracy"])
                    csv_results.append(["MNIST " + experiment_dir[-1:], fold_dir[-2:], train_size_dir[-5:], overlap_dir[-4:], config["accuracy"]])
    print_results(results)
    write_csv(csv_results, CSV_RESULTS)

if __name__ == '__main__':
    main()

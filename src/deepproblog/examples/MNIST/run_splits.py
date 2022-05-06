# Lint as: python3
import os


THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(THIS_DIR, 'models', 'data', 'MNIST', 'psl_data', 'mnist-addition')
RESULTS_DIR = os.path.join(THIS_DIR, 'results')

EVAL_DIR = 'eval'
LEARN_DIR = 'learn'
IMAGESUM_FILENAME = 'imagesum_truth.txt'
CONFIG_FILENAME = 'config.json'


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

    for experiment_dir in os.listdir(DATA_DIR):
        experiment_path = os.path.join(DATA_DIR, experiment_dir)
        experiment_result_path = os.path.join(RESULTS_DIR, experiment_dir)
        os.makedirs(experiment_result_path, exist_ok=True)

        for fold_dir in os.listdir(experiment_path):
            fold_path = os.path.join(experiment_path, fold_dir)
            if os.path.isfile(fold_path):
                continue
            fold_result_path = os.path.join(experiment_result_path, fold_dir)
            os.makedirs(fold_result_path, exist_ok=True)

            for train_size_dir in os.listdir(fold_path):
                train_size_path = os.path.join(fold_path, train_size_dir)
                train_size_result_path = os.path.join(fold_result_path, train_size_dir)
                os.makedirs(train_size_result_path, exist_ok=True)

                for overlap_dir in os.listdir(train_size_path):
                    overlap_path = os.path.join(train_size_path, overlap_dir)
                    overlap_result_path = os.path.join(train_size_result_path, overlap_dir)
                    os.makedirs(overlap_result_path, exist_ok=True)

                    imagesum_learn_path = os.path.join(overlap_path, LEARN_DIR, IMAGESUM_FILENAME)
                    imagesum_eval_path = os.path.join(overlap_path, EVAL_DIR, IMAGESUM_FILENAME)

                    config_path = os.path.join(overlap_result_path, CONFIG_FILENAME)
                    if os.path.isfile(config_path):
                        print("Found existing config file, skipping run: %s" % (overlap_path,))
                        continue

                    train_indicies = load_data(imagesum_learn_path)
                    test_indicies = load_data(imagesum_eval_path)

                    return 0


if __name__ == '__main__':
    main()
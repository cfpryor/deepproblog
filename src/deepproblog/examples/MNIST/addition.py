import sys
import time
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.network import MNIST_Net
from deepproblog.examples.MNIST.data import (
    MNIST_train,
    MNIST_test,
    addition,
)
from deepproblog.heuristics import geometric_mean
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.utils import get_configuration, format_time_precise, \
    config_to_string


def main(method="gm",
         num_digits=1,
         pretrain_val=0,
         exploration=False,
         seed=0,
         num_epochs=1,
         batch_size=2,
         learning_rate=1e-3,
         log_iterations=100,
         train_indices=None,
         test_indices=None,
         train_set_name="train",
         test_set_name="test"):
    # i = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # parameters = {
    #     "method": ["gm", "exact"],
    #     "N": [1, 2, 3],
    #     "pretrain": [0],
    #     "exploration": [False, True],
    #     "run": range(5),
    # }

    # configuration = get_configuration(parameters, i)
    configuration = {
        "method": method,
        "N": num_digits,
        "pretrain": pretrain_val,
        "exploration": exploration,
        "run": seed,
    }
    torch.manual_seed(configuration["run"])

    name = "addition_" + config_to_string(
        configuration) + "_" + format_time_precise()
    print(name)

    train_set = addition(configuration["N"], train_set_name, indices=train_indices)
    test_set = addition(configuration["N"], test_set_name, indices=test_indices)

    network = MNIST_Net()

    pretrain = configuration["pretrain"]
    if pretrain is not None and pretrain > 0:
        network.load_state_dict(
            torch.load("models/pretrained/all_{}.pth".format(
                configuration["pretrain"]))
        )
    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    model = Model("models/addition.pl", [net])
    if configuration["method"] == "exact":
        if configuration["exploration"] or configuration["N"] > 2:
            print("Not supported?")
            exit()
        model.set_engine(ExactEngine(model), cache=True)
    elif configuration["method"] == "gm":
        model.set_engine(
            ApproximateEngine(
                model, 1, geometric_mean,
                exploration=configuration["exploration"]
            )
        )
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    loader = DataLoader(train_set, batch_size, False)
    time_start = time.time()
    train = train_model(model, loader, num_epochs, log_iter=log_iterations, profile=0)
    time_end = time.time()
    model.save_state("snapshot/" + name + ".pth")
    accuracy = get_confusion_matrix(model, test_set, verbose=-1).accuracy()
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment("Accuracy {}".format(accuracy))
    train.logger.write_to_file("log/" + name)

    config = {
        "name": name,
        "train_set_name": train_set_name,
        "test_set_name": test_set_name,
        "method": method,
        "num_digits": num_digits,
        "pretrain_value": pretrain_val,
        "exploration": exploration,
        "seed": seed,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "accuracy": accuracy,
        "learning_time_start": time_start,
        "learning_time_end": time_end,
        "learning_total_time": time_end - time_start
    }

    return config


if __name__ == '__main__':
    main()

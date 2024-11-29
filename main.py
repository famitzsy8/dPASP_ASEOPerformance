import sys
import os
from experiment import Experiment

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <filename> <num_queries> <estimated_models>")
        sys.exit(1)

    file_arg = sys.argv[1]
    num_queries = int(sys.argv[2])
    estimated_models = int(sys.argv[3])

    if not os.path.exists("./experiments/" + file_arg):
        print("Please provide a valid filename")
        sys.exit(1)

    experiment = Experiment(file_arg, num_queries, estimated_models=estimated_models)
    experiment.run()
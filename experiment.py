import os
import shutil
import subprocess
import time
import psutil
import numpy as np
import pickle
from utils import extract_probabilities, generate_nmodels_list
from plotting import plot_results
import sys

class Experiment:
    """
    Class to run probabilistic logic program experiments using the pasp tool.
    """

    def __init__(self, file_arg, num_queries, estimated_models):
        """
        Initialize the Experiment instance.

        Parameters:
        - file_arg (str): Path to the logic program (.lp) file.
        - num_queries (int): Number of query probabilities expected.
        """
        self.file_path = "./experiments/" + file_arg
        self.num_queries = num_queries
        self.name = os.path.splitext(os.path.basename(file_arg))[0]
        # This list will hold all the results we dump into the pickle file 
        # [['exact', probabilities, runtime, memory_usage], [nmodels_1, probabilities, runtime, memory_usage], ...,  [nmodels_N, probabilities, runtime, memory_usage]]
        self.results = []  
        self.skip_exact = False
        self.estimated_models = estimated_models
        self.nmodels_list = generate_nmodels_list(self.estimated_models)

        # Explanation: Most programs took way too long to infer the exact probability. 
        # However in most cases the ASEO algorithm converged to 0 (continuously) after some B_i was reached.
        # This is a clear indication that the program has no more models to consider and by running ASEO with multiple different B_i
        # we found such a B_i for every program which in the end, describes the complexity of the program. We took this B_i as our total model count
        # for the program and the probability of P_aseo(Q_P, B_i) as the exact probability (which is equal to the exact probability). This is why
        # we skip the exact inference for these 5 out of 6 programs, and use the last approximation (= exact probability) as the "surrogate" instead.
        if self.name in ['argumentation', 'mango', 'smoke', 'arithmetic', 'latinsquare']:
            print(f"Detected '{self.name}.lp'. Skipping exact inference.")
            self.skip_exact = True

        # We create the directories for each program/experiment that was run
        self.create_directories()

    def create_directories(self):
        """
        Create directories required for storing results and plots.
        If directories already exist, delete them (except for main_plots).
        """
        try:
            # We don't want to accumulate tons of different experiments, so each time we run it, we delete the last
            if os.path.exists(f"{self.name}"):
                shutil.rmtree(f"{self.name}")
                
            os.makedirs("results", exist_ok=True)
            os.makedirs(f"results/{self.name}", exist_ok=True)
            os.makedirs(f"results/{self.name}/plots", exist_ok=True)
            os.makedirs(f"results/{self.name}/plots/exactVapprox", exist_ok=True)
            os.makedirs(f"results/{self.name}/plots/abs_error", exist_ok=True)
            os.makedirs(f"results/{self.name}/plots/error_vs_nmodels", exist_ok=True)
            os.makedirs(f"results/{self.name}/plots/runtime", exist_ok=True)
            os.makedirs(f"results/{self.name}/plots/spheres", exist_ok=True)

            # This is a directory that accumulates plots from all different experiments
            os.makedirs("main_plots", exist_ok=True)
        except Exception as e:
            print(f"Couldn't make directories: {e}")
            sys.exit(1)

    def run(self):
        """
        Run the experiment by executing exact inference (if not skipped) and approximate inference for various nmodels.
        """
        # If the results file exists already, we don't need to run the inference again, just use the previous results to generate the plots
        results_file = os.path.join("results", f'results_{self.name}.pkl')
        if os.path.exists(results_file):
            print(f"Results for '{self.name}' already exist. Loading from file.")
            with open(results_file, 'rb') as f:
                self.results, self.nmodels_list = pickle.load(f)
        else:
            if not self.skip_exact:
                print("Now running exact inference")
                self.run_exact()
                print("Ran exact inference")

            for nmodels in self.nmodels_list:
                self.modify_and_run_lp_file(nmodels)

            # Here we save the results and the list of nmodels we used (and is different for every experiment based on the self.estimated_models parameter)
            with open(results_file, 'wb') as f:
                pickle.dump((self.results, self.nmodels_list), f)

        plot_results(self)
        print(f"Plots saved in results/{self.name}/plots/")

    def run_exact(self):
        """
        Run exact inference using the pasp tool and collect results.
        """
        try:
            # here we create a local copy into the directory we specifically created for our experiment
            copy_path = f".results/{self.name}/test_exact.lp"
            shutil.copy2(self.file_path, copy_path)

            with open(copy_path, 'a') as file:
                file.write('\n#inference exact.\n')

            print(f"Running {copy_path} (exact inference)")

            # Measuring the runtime 
            start_time = time.perf_counter()

            # We spawn our inference as a subprocess from the one launched by Python
            process = subprocess.Popen(['pasp', copy_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pid = process.pid
            ps_process = psutil.Process(pid)
            max_memory = 0  # in bytes

            while True:
                retcode = process.poll()
                try:
                    mem = ps_process.memory_info().rss
                    if mem > max_memory:
                        max_memory = mem
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if retcode is not None:
                    break

            stdout, stderr = process.communicate()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time  # in seconds
            max_memory_mb = max_memory / (1024 * 1024)  # Convert bytes to MB

            probabilities = extract_probabilities(stdout + '\n' + stderr, self.num_queries)

            self.results.insert(0, ('exact', probabilities, elapsed_time, max_memory_mb))

        except Exception as e:
            print(f"An error occurred during exact inference: {e}")

    def modify_and_run_lp_file(self, nmodels):
        """
        Modify the LP file to include approximate inference with given nmodels, run it, and collect results.

        Parameters:
        - nmodels (int): Number of models to use for approximate inference.
        """
        try:
            # Again, we copy the file to our local directory so we can then modify it
            copy_path = f"./results/{self.name}/test_{nmodels}.lp"
            shutil.copy2(self.file_path, copy_path)

            # Here we modify the copied file so that it runs aseo with our specified number of nmodels
            with open(copy_path, 'a') as file:
                file.write(f'\n#inference aseo, nmodels={nmodels}.\n')

            print(f"Running {copy_path} with nmodels={nmodels}")

            start_time = time.perf_counter()

            process = subprocess.Popen(['pasp', copy_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pid = process.pid
            ps_process = psutil.Process(pid)
            max_memory = 0

            while True:
                retcode = process.poll()
                try:
                    mem = ps_process.memory_info().rss
                    if mem > max_memory:
                        max_memory = mem
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if retcode is not None:
                    break

            stdout, stderr = process.communicate()
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            max_memory_mb = max_memory / (1024 * 1024)

            probabilities = extract_probabilities(stdout + '\n' + stderr, self.num_queries)

            self.results.append((nmodels, probabilities, elapsed_time, max_memory_mb))

        except Exception as e:
            print(f"An error occurred while running nmodels={nmodels}: {e}")

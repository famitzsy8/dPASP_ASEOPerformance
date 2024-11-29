# All the plotting functions reside in this file

# We generate the following plots:
#
# - Absolute Error vs. model bounds (B) --> plots/abs_error
# - E_metric vs. B --> plots/error_vs_nmodels
# - Approximated Probability vs. Exact Probability of Query --> plots/exactVapprox
# - Runtime vs. B --> plots/runtime
# - Number of Models needed to guarantee E_metric < k --> plots/spheres

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

def plot_results(experiment):
    """
    Plot the results, including probabilities vs nmodels, error plots, runtime plot,
    and the spheres plot representing convergence thresholds.
    """
    self = experiment # For convenience resasons

    # Separate the exact results and the approximations
    exact_probabilities = None
    approx_results = []
    for item in self.results:
        if item[0] == 'exact':
            exact_probabilities = item[1]
        else:
            approx_results.append(item)

    nmodels_list = [item[0] for item in approx_results]
    runtimes = [item[2] for item in approx_results]

    # If exact probabilities are not available, we use the last approximate probabilities as our 'exact' values
    # We can then use these "best approximations" to compare our other approximations
    if exact_probabilities is None:
        if approx_results:
            exact_probabilities = approx_results[-1][1]
            print(f"Using last approximate probabilities as 'exact' values: {exact_probabilities}")
            self.skip_convergence = False
        else:
            print("No approximate results available to use as 'exact' values.")
            return
    else:
        self.skip_convergence = False


    for i in range(self.num_queries):
        probs = [item[1][i] if i < len(item[1]) else float('nan') for item in approx_results]

        plt.figure(figsize=(10, 6))
        plt.plot(nmodels_list, probs, marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='#1f77b4', label=f'Approximate Probability')
        if i < len(exact_probabilities):
            plt.axhline(y=exact_probabilities[i], color='r', linestyle='--', linewidth=2,
                        label='Exact Probability')
        plt.xscale('log') # We log-scale the x-axis to distribute our samples more equidistantly
        plt.xlabel('Number of Models (nmodels)', fontsize=14, fontweight='bold')
        plt.ylabel('Probability', fontsize=14, fontweight='bold')
        plt.title(f'Probability vs. nmodels for Query {i+1}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/plots/exactVapprox/query_{i+1}.png', dpi=300)
        plt.savefig(f'main_plots/{self.name}_eva{i}', dpi=300)
        plt.close()

        # Computation & Plotting of the absolute error
        exact_prob = exact_probabilities[i] if i < len(exact_probabilities) else float('nan')
        abs_errors = [abs(exact_prob - p) if not np.isnan(p) and not np.isnan(exact_prob) else np.nan for p in probs]

        plt.figure(figsize=(10, 6))
        plt.plot(nmodels_list, abs_errors, marker='s', linestyle='-', linewidth=2, markersize=6,
                 color='#ff7f0e', label='Absolute Error')
        plt.xscale('log')
        plt.xlabel('Number of Models (nmodels)', fontsize=14, fontweight='bold')
        plt.ylabel('Absolute Error', fontsize=14, fontweight='bold')
        plt.title(f'Absolute Error vs. nmodels for Query {i+1}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/plots/abs_error/query_{i+1}.png', dpi=300)
        plt.close()

        # Computation of E_metric
        maximum_error = max([abs(exact_prob - p) for p in probs if not np.isnan(p)])
        first_error = abs(exact_prob - probs[0])
        reference_error = (maximum_error + first_error) / 2

        rel_errors = [
            abs(exact_prob - p) / (max(0.01, reference_error)) if not np.isnan(p)
            and not np.isnan(exact_prob) and exact_prob != 0 else np.nan
            for p in probs
        ] # max(0.01, X) in order to avoid numerical instability

        # Plotting of E_metric vs. B
        plt.figure(figsize=(10, 6))
        plt.plot(nmodels_list, rel_errors, marker='o', linestyle='-', linewidth=2, markersize=6,
                 color='#9467bd', label='Relative Error')
        plt.xscale('log')
        plt.xlabel('Number of Models (nmodels)', fontsize=14, fontweight='bold')
        plt.ylabel('Relative Error', fontsize=14, fontweight='bold')
        plt.title(f'Relative Error vs. nmodels for Query {i+1}', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/plots/error_vs_nmodels/query_{i+1}.png', dpi=300)
        plt.savefig(f'main_plots/{self.name}_rel{i}', dpi=300)
        plt.close()

        # Computation of AUEC using the E_metric vs nmodels (B) curve
        # Here we calculate the AUEC values for each query
        auec_list = []
        convergence_points = {}  # A dictionary to store the nmodels after which the error falls below a certain threshold
        
        nmodels_array = np.array(nmodels_list)
        rel_errors_array = np.array(rel_errors)

        # Detection & Removal of NaN values
        valid_indices = ~np.isnan(rel_errors_array)
        nmodels_valid = nmodels_array[valid_indices]
        nmodels_valid_log = np.log(nmodels_valid) # Log-scaling of the nmodels_array
        rel_errors_valid = rel_errors_array[valid_indices]
        rel_errors_valid = np.log((rel_errors_valid * (math.exp(1) - 1)) + 1) # Re-scaling of E_metric

        # Compute AUEC
        if len(nmodels_valid) >= 2:
            auec = np.trapz(rel_errors_valid, x=nmodels_valid_log)
            print(f"AUEC (Relative Error vs nmodels) for Query {i+1}: {auec}")
            auec_list.append(auec)
        else:
            auec = np.nan
            print(f"AUEC (Relative Error vs nmodels) for Query {i+1} cannot be computed due to insufficient data.")
            auec_list.append(auec)

        # Calculating the nmodels needed for ASEO to constantly stay below a certain error margin/threshold
        thresholds = [0.2, 0.08, 0.03, 0.01, 0.008, 0.005, 0.002, 0.001, 0.0002] # Our pre-selected set of thresholds
        convergence_points[i] = {}
        for threshold in thresholds:
            idx = min([index for index, value in enumerate(rel_errors)
                        if not np.isnan(value) and value < threshold and max(rel_errors[index:]) < threshold])
            if idx is not None:
                convergence_nmodels = nmodels_list[idx]
                convergence_points[i][threshold] = convergence_nmodels
                print(f"Query {i+1} reached relative error < {threshold} at nmodels = {convergence_nmodels}")
            else:
                convergence_points[i][threshold] = None
                print(f"Query {i+1} did not reach relative error < {threshold}")
                
        # Here we plot the obtained nmodels from the error margin calculations above and plot their size by creating a sphere of size nmodels
        total_models = self.estimated_models
        thresholds = [0.2, 0.08, 0.03, 0.01, 0.008, 0.005, 0.002, 0.001, 0.0002]
        threshold_labels = [f'< {t}' for t in thresholds] + ['Total']

        convergence_nmodels = [convergence_points[i].get(t, total_models) for t in thresholds] + [total_models]

        sizes = [n / total_models * 8000 for n in convergence_nmodels]  # Adjusted scaling factor

        # Define color palette for our spheres
        colors = [
            "#e3f2fd",  # Very light blue
            "#bbdefb",  # Light blue
            "#90caf9",  # Medium light blue
            "#64b5f6",  # Moderate blue
            "#42a5f5",  # Blue
            "#2196f3",  # Standard blue
            "#1e88e5",  # Medium dark blue
            "#1976d2",  # Dark blue
            "#0d47a1",  # Very dark blue
            "#000000"   # Black
        ]
        # Ensure all lists have the same length
        assert len(threshold_labels) == len(convergence_nmodels) == len(sizes) == len(colors), "List lengths are mismatched."

        # Crafting the legend to display the value of nmodels next to the spheres
        legend_labels = [f'{label} ({n} models)' for label, n in zip(threshold_labels, convergence_nmodels)]
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=8)
            for label, color in zip(legend_labels, colors)
        ]

        # We create two plots: One with the spheres (representing the nmodels size for a certain threshold) overlapping, and one with them side-by-side
        
        # Overlapping
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        ax.set_aspect('equal')
        max_size = max(sizes)
        max_radius = np.sqrt(max_size) * 1.3  # 1.3 is a visual scaling factor open to change
        for j in reversed(range(len(sizes))):
            radius = np.sqrt(sizes[j])
            circle = Circle((0, radius), radius, color=colors[j], alpha=0.5)
            ax.add_patch(circle)
        ax.set_xlim(-max_radius * 1.1, max_radius * 1.1)
        ax.set_ylim(0, max_radius * 2.2)
        plt.title(f'Convergence Spheres (Overlapping) for {self.name}', fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.legend(handles=legend_elements, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/plots/spheres/query_{i+1}_overlapping.png', dpi=300)
        plt.close()

        # Side by-side
        plt.figure(figsize=(10, 3))
        ax = plt.gca()
        ax.set_aspect('equal')
        max_size = max(sizes)
        max_radius = np.sqrt(max_size)
        spacing = max_radius * 2.5
        x_positions = np.arange(len(sizes)) * spacing
        for j in range(len(sizes)):
            radius = np.sqrt(sizes[j])
            circle = Circle((x_positions[j], radius), radius, color=colors[j], alpha=0.9)
            ax.add_patch(circle)
        ax.set_xlim(-max_radius, x_positions[-1] + max_radius * 1.5)
        ax.set_ylim(0, max_radius * 2.2)
        plt.xticks(x_positions, threshold_labels, fontsize=10)
        plt.yticks([])
        plt.title(f'Convergence Spheres (Side by Side) for {self.name}', fontsize=14, fontweight='bold')
        plt.legend(handles=legend_elements, loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'results/{self.name}/plots/spheres/query_{i+1}_side_by_side.png', dpi=300)
        plt.close()
        
        
    # Independently of the number of queries we plot the runtimes needed
    plt.figure(figsize=(8, 5))
    plt.plot(nmodels_list, runtimes, marker='o', linestyle='-', linewidth=2, markersize=6,
            color='#d62728', label='Runtime')
    plt.xlabel('Number of Models (nmodels)', fontsize=12, fontweight='bold')
    plt.ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    plt.title('Runtime vs. nmodels', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'results/{self.name}/plots/runtime/runtime_vs_nmodels.png', dpi=300)
    plt.close()
        
    # And eventually print out the AUEC values to the console
    print("\nAUEC values for each query:")
    for i, auec in enumerate(auec_list):
        print(f"Query {i+1}: AUEC (Relative Error vs nmodels) = {auec}")

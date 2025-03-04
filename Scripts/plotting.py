import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def main(results_path="results.csv", prefix=""):
    # Load the CSV data
    df = pd.read_csv(results_path).drop(index=[0,1,2,3,4]).reset_index(drop=True)

    # Plot settings
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Plot each optimization method
    for col in df.columns[1:]:  # Skip 'step' column
        plt.plot(df['step'], df[col], linestyle='-', label=col)

    # Labels and legend
    plt.xlabel("Step")
    plt.ylabel("Performance")
    plt.title("Performance Over Steps for Different Optimizations")
    plt.legend()

    # Save plot before showing
    plt.savefig(f'{prefix}optimization_performance_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Set the step as index
    df_heatmap = df.set_index("step")

    # Plot heatmap (Reversed color since lower time is better)
    plt.figure(figsize=(12, 6))
    sns.heatmap(df_heatmap.T, cmap="viridis_r", annot=False, linewidths=0.5)

    # Labels and title
    plt.xlabel("Step")
    plt.ylabel("Optimization Type")
    plt.title("Performance Heatmap (Lower is Better)")

    # Save plot with descriptive name
    plt.savefig(f'{prefix}performance_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Compute mean execution time for each optimization method
    mean_times = df.iloc[:, 1:].mean()

    # Compute overall speedup (using mean execution times)
    overall_speedup = mean_times["no_optimization"] / mean_times.drop("no_optimization")

    # Plot mean relative speedup as a horizontal bar chart
    plt.figure(figsize=(10, 6))
    overall_speedup.plot(kind="barh", color="skyblue", edgecolor="black")

    # Formatting
    plt.xlabel("Mean Relative Speedup")
    plt.ylabel("Optimization Method")
    plt.title("Mean Relative Speedup Compared to No Optimization")
    plt.axvline(x=1, color="red", linestyle="--", label="Baseline (No Speedup)")
    plt.legend()
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Save plot with descriptive name
    plt.savefig(f'{prefix}mean_speedup_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate optimization comparison plots')
    parser.add_argument('--prefix', type=str, default="", help='Prefix for output filenames')
    parser.add_argument('--results_path', type=str, default="results.csv", help='A path for results CSV file')
    args = parser.parse_args()
    main(args.results_path, args.prefix)
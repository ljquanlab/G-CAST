# app.py
import argparse
from GCAST_Runner import GCAST_Runner

def main():
    parser = argparse.ArgumentParser(description="Run GCAST")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input data (.h5 or folder)")
    parser.add_argument("--sample_name", type=str, default="sample", help="Sample name")
    parser.add_argument("--n_clusters", type=int, required=True, default=7, help="Number of clusters")


    # parser.add_argument("--tech", type=str, default="10xvisium_raw", help="Technology type: 10xvisium_raw or other")
    parser.add_argument("--label", type=bool, default=True, help="Whether ground truth labels are available")
    parser.add_argument("--negi", type=int, default=12, help="Number of neighbors for graph construction")
    parser.add_argument("--mode", type=str, default="single", help="Mode: single or batch")
    parser.add_argument("--cluster_method", type=str, default="mclust", help="Clustering method: KMeans, KMedoids, leiden, louvain, mclust")
    parser.add_argument("--random_seed", type=int, default=2025, help="Random seed")


    # parser.add_argument("--output", type=str, default="./output", help="Path to save results")
    args = parser.parse_args()

    # print("🚀 Running GCAST ...")
    GCAST_Runner(path = args.input_path,
                n_clusters=args.n_clusters,
                sample_name=args.sample_name,
                label=args.label,
                negi=args.negi,
                mode=args.mode,
                cluster_method=args.cluster_method,
                random_seed=args.random_seed,
                 )
    print(f"DONE!")


if __name__ == "__main__":
    main()

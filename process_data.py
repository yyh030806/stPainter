import os
import argparse
from src.data import process_sc, calculate_gene_sparsity, process_st

from src.parsing import parse_data_args

def main(args):
    os.makedirs(args.processed_data_dir, exist_ok=True)

    # Define standard file paths
    sc_raw_file_path = os.path.join(args.raw_data_dir, "sc_raw.h5ad")
    sc_processed_file_path = os.path.join(args.processed_data_dir, "sc_train.h5ad")
    gene_sparsity_ratio_file_path = os.path.join(args.processed_data_dir, "gene_sparsity_ratio.csv")

    # Processing SC Data
    if os.path.exists(sc_raw_file_path):
        print(f"\nFound SC raw data: {sc_raw_file_path}")
        
        # Preprocess SC
        process_sc(sc_raw_file_path, sc_processed_file_path)
        
        # Calculate sparsity
        print("Calculating gene sparsity...")
        calculate_gene_sparsity(sc_processed_file_path, gene_sparsity_ratio_file_path)
        
    elif os.path.exists(sc_processed_file_path):
        print(f"\nSC raw data not found, using existing processed file: {sc_processed_file_path}")
    else:
        print("\nWarning: No SC data found (raw or processed). Skipping SC step.")

    # Processing ST Data
    # ST processing requires the processed SC file as a reference
    if os.path.exists(sc_processed_file_path):
        st_raw_files = [f for f in os.listdir(args.raw_data_dir) if f.startswith("st_") and f.endswith("_raw.h5ad")]

        if st_raw_files:
            print(f"\nFound {len(st_raw_files)} ST raw file(s). Processing...")
            
            for st_raw_file_name in st_raw_files:
                st_raw_full_path = os.path.join(args.raw_data_dir, st_raw_file_name)
                
                # Construct output filename
                base_name = os.path.splitext(st_raw_file_name)[0]
                processed_file_name = base_name.replace("_raw", "_test") + ".h5ad"
                st_processed_full_path = os.path.join(args.processed_data_dir, processed_file_name)

                print(f"Processing: {st_raw_full_path}")
                process_st(sc_processed_file_path, st_raw_full_path, st_processed_full_path)
                print(f"Saved: {st_processed_full_path}")
        else:
            print("\nNo ST raw files found.")
    else:
        print("\nSkipping ST processing: Processed SC file required but not found.")
    
    print("\nPipeline completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parse_data_args(parser)

    args = parser.parse_args()
    main(args)
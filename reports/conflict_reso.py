import os
import pandas as pd
import glob
from pathlib import Path

def main():
    """
    Main function to analyze SIV signal coherence and conflict resolution accuracy.
    """
    # Get all CSV files from pfpa_input folder
    reports_dir = Path(__file__).parent
    input_dir = reports_dir / "pfpa_input"
    csv_files = glob.glob(str(input_dir / "backtest_*.csv"))
    
    if not csv_files:
        print("No CSV files found in pfpa_input directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Combine all CSV files
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Keep only rows with SIV_Signal and Correct columns
            if 'SIV_Signal' in df.columns and 'Correct' in df.columns:
                df_valid = df[['SIV_Signal', 'Correct']].dropna()
                all_data.append(df_valid)
                print(f"Processed: {os.path.basename(csv_file)} ({len(df_valid)} valid rows)")
            else:
                print(f"Skipped: {os.path.basename(csv_file)} (missing SIV_Signal or Correct columns)")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_data:
        print("No valid data found in CSV files.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined_df)}")
    
    # Case 1: COHERENT Accuracy
    coherent_df = combined_df[combined_df['SIV_Signal'] == 'COHERENT']
    coherent_correct = len(coherent_df[coherent_df['Correct'] == True])
    coherent_total = len(coherent_df)
    coherent_accuracy = coherent_correct / coherent_total if coherent_total > 0 else 0
    
    # Case 2: PARTIAL (CONFLICT) Accuracy
    partial_df = combined_df[combined_df['SIV_Signal'] == 'PARTIAL']
    partial_correct = len(partial_df[partial_df['Correct'] == True])
    partial_total = len(partial_df)
    partial_accuracy = partial_correct / partial_total if partial_total > 0 else 0
    
    # Create results
    results = [
        {
            'Case': 'COHERENT',
            'Correct': coherent_correct,
            'Total': coherent_total,
            'Accuracy_Fraction': f"{coherent_correct}/{coherent_total}" if coherent_total > 0 else "0/0",
            'Accuracy_Percentage': f"{coherent_accuracy * 100:.2f}%" if coherent_total > 0 else "N/A"
        },
        {
            'Case': 'PARTIAL',
            'Correct': partial_correct,
            'Total': partial_total,
            'Accuracy_Fraction': f"{partial_correct}/{partial_total}" if partial_total > 0 else "0/0",
            'Accuracy_Percentage': f"{partial_accuracy * 100:.2f}%" if partial_total > 0 else "N/A"
        }
    ]
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV in pfpa_input folder
    output_dir = input_dir
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "conflict_resolution_metrics.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Display results
    print(results_df.to_string(index=False))
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("DETAILED SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nCASE 1: COHERENT ACCURACY")
    print(f"  Total COHERENT cases: {coherent_total}")
    print(f"  COHERENT cases marked as True: {coherent_correct}")
    print(f"  Accuracy: {coherent_correct}/{coherent_total} = {coherent_accuracy * 100:.2f}%")
    
    print(f"\nCASE 2: PARTIAL (CONFLICT) ACCURACY")
    print(f"  Total PARTIAL cases: {partial_total}")
    print(f"  PARTIAL cases marked as True: {partial_correct}")
    print(f"  Accuracy: {partial_correct}/{partial_total} = {partial_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

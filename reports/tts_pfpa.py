import os
import pandas as pd
import glob
from pathlib import Path

def extract_tts_signal(final_verdict):
    """
    Extract TTS signal from Final_Verdict column.
    Normalize to BUY, SELL, or HOLD.
    """
    if pd.isna(final_verdict):
        return None
    
    verdict_str = str(final_verdict).strip().upper()
    
    # Map various verdict types to BUY, SELL, HOLD
    if verdict_str in ['BUY', 'BULLISH']:
        return 'BUY'
    elif verdict_str in ['SELL', 'BEARISH']:
        return 'SELL'
    elif verdict_str in ['HOLD', 'NEUTRAL']:
        return 'HOLD'
    else:
        return None


def calculate_metrics(signal_type, df):
    """
    Calculate precision, recall, and F1 score for a specific signal type.
    
    Args:
        signal_type: 'BUY', 'SELL', or 'HOLD'
        df: DataFrame with 'TTS_Signal' and 'Correct' columns
    
    Returns:
        Dictionary with TP, FP, FN, Precision, Recall, F1
    """
    # Count True Positives: Signal matches AND Correct is True
    tp = len(df[(df['TTS_Signal'] == signal_type) & (df['Correct'] == True)])
    
    # Count False Positives: Signal matches AND Correct is False
    fp = len(df[(df['TTS_Signal'] == signal_type) & (df['Correct'] == False)])
    
    # Count False Negatives: Signal doesn't match AND Correct is True
    fn = len(df[(df['TTS_Signal'] != signal_type) & (df['Correct'] == True)])
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Signal': signal_type,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision_Fraction': f"{tp}/{tp + fp}" if (tp + fp) > 0 else "0/0",
        'Precision_Percentage': f"{precision * 100:.2f}%" if (tp + fp) > 0 else "N/A",
        'Recall_Fraction': f"{tp}/{tp + fn}" if (tp + fn) > 0 else "0/0",
        'Recall_Percentage': f"{recall * 100:.2f}%" if (tp + fn) > 0 else "N/A",
        'F1_Score': f"{f1:.4f}",
        'Precision_Value': precision,
        'Recall_Value': recall,
        'F1_Value': f1
    }


def main():
    """
    Main function to process all CSV files and generate metrics.
    """
    # Get all CSV files from pfpa_input folder
    reports_dir = Path(__file__).parent
    input_dir = reports_dir / "pfpa_input"
    csv_files = glob.glob(str(input_dir / "backtest_*.csv"))
    
    if not csv_files:
        print("No CSV files found in reports directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    
    # Combine all CSV files
    all_data = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Extract TTS signal from Final_Verdict
            df['TTS_Signal'] = df['Final_Verdict'].apply(extract_tts_signal)
            # Keep only valid signals
            df_valid = df[df['TTS_Signal'].notna()].copy()
            all_data.append(df_valid)
            print(f"Processed: {os.path.basename(csv_file)} ({len(df_valid)} valid rows)")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_data:
        print("No valid data found in CSV files.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined_df)}")
    
    # Calculate metrics for each signal type
    metrics_list = []
    for signal in ['BUY', 'SELL', 'HOLD']:
        metrics = calculate_metrics(signal, combined_df)
        metrics_list.append(metrics)
    
    # Calculate overall accuracy
    total_correct = len(combined_df[combined_df['Correct'] == True])
    total_samples = len(combined_df)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Create results DataFrame
    results_df = pd.DataFrame(metrics_list)
    
    # Select columns for output in the requested format
    output_df = results_df[[
        'Signal',
        'TP',
        'FP',
        'FN',
        'Precision_Fraction',
        'Precision_Percentage',
        'Recall_Fraction',
        'Recall_Percentage',
        'F1_Score'
    ]].copy()
    
    # Add overall accuracy row
    accuracy_row = pd.DataFrame([{
        'Signal': 'OVERALL',
        'TP': f"{total_correct}/{total_samples}",
        'FP': '',
        'FN': '',
        'Precision_Fraction': '',
        'Precision_Percentage': '',
        'Recall_Fraction': '',
        'Recall_Percentage': '',
        'F1_Score': f"{overall_accuracy * 100:.2f}%"
    }])
    output_df = pd.concat([output_df, accuracy_row], ignore_index=True)
    
    # Save to CSV in pfpa_input folder
    output_dir = input_dir
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "tts_metrics.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Display results
    print(output_df.to_string(index=False))
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    for _, row in results_df.iterrows():
        signal = row['Signal']
        print(f"\n{signal} Signal:")
        print(f"  TP: {row['TP']}, FP: {row['FP']}, FN: {row['FN']}")
        print(f"  Precision: {row['Precision_Fraction']} = {row['Precision_Percentage']}")
        print(f"  Recall: {row['Recall_Fraction']} = {row['Recall_Percentage']}")
        print(f"  F1 Score: {row['F1_Score']}")
    
    # Print overall accuracy
    print(f"\n{'='*80}")
    print("OVERALL ACCURACY")
    print(f"{'='*80}")
    print(f"Accuracy: {total_correct}/{total_samples} = {overall_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

import os
import pandas as pd
import glob
from pathlib import Path

def extract_verdict(final_verdict):
    """
    Extract verdict from Final_Verdict column.
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


def main():
    """
    Main function to analyze MAS accuracy and Buy+Sell accuracy.
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
            # Keep only rows with Final_Verdict and Correct columns
            if 'Final_Verdict' in df.columns and 'Correct' in df.columns:
                # Extract verdict
                df['Verdict'] = df['Final_Verdict'].apply(extract_verdict)
                # Keep only valid verdicts
                df_valid = df[df['Verdict'].notna()][['Verdict', 'Correct']].copy()
                all_data.append(df_valid)
                print(f"Processed: {os.path.basename(csv_file)} ({len(df_valid)} valid rows)")
            else:
                print(f"Skipped: {os.path.basename(csv_file)} (missing Final_Verdict or Correct columns)")
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    if not all_data:
        print("No valid data found in CSV files.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined_df)}")
    
    # MAS Accuracy: (total number of "True")/(total number of BUY+SELL+HOLD)
    mas_total = len(combined_df)
    mas_correct = len(combined_df[combined_df['Correct'] == True])
    mas_accuracy = mas_correct / mas_total if mas_total > 0 else 0
    
    # Buy+Sell Accuracy: (total number of "True")/(total number of BUY+SELL)
    buy_sell_df = combined_df[combined_df['Verdict'].isin(['BUY', 'SELL'])]
    buy_sell_total = len(buy_sell_df)
    buy_sell_correct = len(buy_sell_df[buy_sell_df['Correct'] == True])
    buy_sell_accuracy = buy_sell_correct / buy_sell_total if buy_sell_total > 0 else 0
    
    # Get breakdown by verdict type
    buy_df = combined_df[combined_df['Verdict'] == 'BUY']
    sell_df = combined_df[combined_df['Verdict'] == 'SELL']
    hold_df = combined_df[combined_df['Verdict'] == 'HOLD']
    
    buy_correct = len(buy_df[buy_df['Correct'] == True])
    sell_correct = len(sell_df[sell_df['Correct'] == True])
    hold_correct = len(hold_df[hold_df['Correct'] == True])
    
    # Create results
    results = [
        {
            'Metric': 'MAS Accuracy',
            'Correct': mas_correct,
            'Total': mas_total,
            'Accuracy_Fraction': f"{mas_correct}/{mas_total}" if mas_total > 0 else "0/0",
            'Accuracy_Percentage': f"{mas_accuracy * 100:.2f}%" if mas_total > 0 else "N/A"
        },
        {
            'Metric': 'Buy+Sell Accuracy',
            'Correct': buy_sell_correct,
            'Total': buy_sell_total,
            'Accuracy_Fraction': f"{buy_sell_correct}/{buy_sell_total}" if buy_sell_total > 0 else "0/0",
            'Accuracy_Percentage': f"{buy_sell_accuracy * 100:.2f}%" if buy_sell_total > 0 else "N/A"
        }
    ]
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV in pfpa_input folder
    output_dir = input_dir
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "mas_accuracy_metrics.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Display results
    print(results_df.to_string(index=False))
    
    # Print detailed summary
    print(f"\n{'='*80}")
    print("DETAILED SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nMAS ACCURACY (All Verdicts)")
    print(f"  Total verdicts (BUY+SELL+HOLD): {mas_total}")
    print(f"  Correct predictions: {mas_correct}")
    print(f"  Accuracy: {mas_correct}/{mas_total} = {mas_accuracy * 100:.2f}%")
    
    print(f"\nBUY+SELL ACCURACY")
    print(f"  Total BUY+SELL verdicts: {buy_sell_total}")
    print(f"  Correct predictions: {buy_sell_correct}")
    print(f"  Accuracy: {buy_sell_correct}/{buy_sell_total} = {buy_sell_accuracy * 100:.2f}%")
    
    print(f"\nBREAKDOWN BY VERDICT TYPE")
    print(f"  BUY - Total: {len(buy_df)}, Correct: {buy_correct}, Accuracy: {buy_correct/len(buy_df)*100:.2f}%" if len(buy_df) > 0 else "  BUY - No data")
    print(f"  SELL - Total: {len(sell_df)}, Correct: {sell_correct}, Accuracy: {sell_correct/len(sell_df)*100:.2f}%" if len(sell_df) > 0 else "  SELL - No data")
    print(f"  HOLD - Total: {len(hold_df)}, Correct: {hold_correct}, Accuracy: {hold_correct/len(hold_df)*100:.2f}%" if len(hold_df) > 0 else "  HOLD - No data")


if __name__ == "__main__":
    main()

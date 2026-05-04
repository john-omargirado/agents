import os
import pandas as pd
import glob
from pathlib import Path

def extract_ce_sentiment(sentiment):
    """
    Extract CE sentiment from Sentiment column.
    Normalize to BULLISH, BEARISH, or NEUTRAL.
    """
    if pd.isna(sentiment):
        return None
    
    sentiment_str = str(sentiment).strip().upper()
    
    # Map various sentiment types to BULLISH, BEARISH, NEUTRAL
    if sentiment_str in ['BULLISH', 'BUY']:
        return 'BULLISH'
    elif sentiment_str in ['BEARISH', 'SELL']:
        return 'BEARISH'
    elif sentiment_str in ['NEUTRAL', 'HOLD']:
        return 'NEUTRAL'
    else:
        return None


def extract_market_reality(market_reality):
    """
    Extract market reality from Market_Reality column.
    Normalize to BULLISH, BEARISH, or NEUTRAL.
    """
    if pd.isna(market_reality):
        return None
    
    reality_str = str(market_reality).strip().upper()
    
    # Map various reality types to BULLISH, BEARISH, NEUTRAL
    if reality_str in ['BULLISH', 'BUY']:
        return 'BULLISH'
    elif reality_str in ['BEARISH', 'SELL']:
        return 'BEARISH'
    elif reality_str in ['NEUTRAL', 'HOLD']:
        return 'NEUTRAL'
    else:
        return None


def calculate_metrics(sentiment_type, df):
    """
    Calculate precision, recall, and F1 score for a specific sentiment type.
    
    Args:
        sentiment_type: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        df: DataFrame with 'CE_Sentiment' and 'Market_Reality' columns
    
    Returns:
        Dictionary with TP, FP, FN, Precision, Recall, F1
    """
    # Count True Positives: Sentiment matches AND Market_Reality matches
    tp = len(df[(df['CE_Sentiment'] == sentiment_type) & (df['Market_Reality'] == sentiment_type)])
    
    # Count False Positives: Sentiment matches AND Market_Reality doesn't match
    fp = len(df[(df['CE_Sentiment'] == sentiment_type) & (df['Market_Reality'] != sentiment_type)])
    
    # Count False Negatives: Sentiment doesn't match AND Market_Reality matches
    fn = len(df[(df['CE_Sentiment'] != sentiment_type) & (df['Market_Reality'] == sentiment_type)])
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Sentiment': sentiment_type,
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
    Main function to process all CSV files and generate CE agent metrics.
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
            
            # Handle both old and new column formats
            if 'Sentiment' in df.columns and 'CE_Sentiment' not in df.columns:
                # Old format: use Sentiment column
                df['CE_Sentiment'] = df['Sentiment'].apply(extract_ce_sentiment)
            elif 'CE_Sentiment' in df.columns:
                # New format: normalize existing CE_Sentiment column
                df['CE_Sentiment'] = df['CE_Sentiment'].apply(extract_ce_sentiment)
            else:
                print(f"Skipped: {os.path.basename(csv_file)} (no Sentiment or CE_Sentiment column)")
                continue
            
            # Extract Market Reality
            df['Market_Reality'] = df['Market_Reality'].apply(extract_market_reality)
            # Keep only valid sentiments
            df_valid = df[df['CE_Sentiment'].notna() & df['Market_Reality'].notna()].copy()
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
    
    # Calculate metrics for each sentiment type
    metrics_list = []
    for sentiment in ['BULLISH', 'BEARISH', 'NEUTRAL']:
        metrics = calculate_metrics(sentiment, combined_df)
        metrics_list.append(metrics)
    
    # Calculate overall accuracy
    total_correct = len(combined_df[combined_df['CE_Sentiment'] == combined_df['Market_Reality']])
    total_samples = len(combined_df)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    # Calculate Bullish+Bearish accuracy
    bullish_bearish_df = combined_df[combined_df['CE_Sentiment'].isin(['BULLISH', 'BEARISH'])]
    bullish_bearish_correct = len(bullish_bearish_df[bullish_bearish_df['CE_Sentiment'] == bullish_bearish_df['Market_Reality']])
    bullish_bearish_total = len(bullish_bearish_df)
    bullish_bearish_accuracy = bullish_bearish_correct / bullish_bearish_total if bullish_bearish_total > 0 else 0
    
    # Create results DataFrame
    results_df = pd.DataFrame(metrics_list)
    
    # Select columns for output in the requested format
    output_df = results_df[[
        'Sentiment',
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
        'Sentiment': 'OVERALL',
        'TP': f"{total_correct}/{total_samples}",
        'FP': '',
        'FN': '',
        'Precision_Fraction': '',
        'Precision_Percentage': '',
        'Recall_Fraction': '',
        'Recall_Percentage': '',
        'F1_Score': f"{overall_accuracy * 100:.2f}%"
    },
    {
        'Sentiment': 'BULLISH+BEARISH',
        'TP': f"{bullish_bearish_correct}/{bullish_bearish_total}",
        'FP': '',
        'FN': '',
        'Precision_Fraction': '',
        'Precision_Percentage': '',
        'Recall_Fraction': '',
        'Recall_Percentage': '',
        'F1_Score': f"{bullish_bearish_accuracy * 100:.2f}%"
    }])
    output_df = pd.concat([output_df, accuracy_row], ignore_index=True)
    
    # Save to CSV in pfpa_input folder
    output_dir = input_dir
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "ce_metrics.csv"
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
        sentiment = row['Sentiment']
        print(f"\n{sentiment} Sentiment:")
        print(f"  TP: {row['TP']}, FP: {row['FP']}, FN: {row['FN']}")
        print(f"  Precision: {row['Precision_Fraction']} = {row['Precision_Percentage']}")
        print(f"  Recall: {row['Recall_Fraction']} = {row['Recall_Percentage']}")
        print(f"  F1 Score: {row['F1_Score']}")
    
    # Print overall accuracy
    print(f"\n{'='*80}")
    print("OVERALL ACCURACY")
    print(f"{'='*80}")
    print(f"Accuracy: {total_correct}/{total_samples} = {overall_accuracy * 100:.2f}%")
    
    print(f"\nBULLISH+BEARISH ACCURACY")
    print(f"Accuracy: {bullish_bearish_correct}/{bullish_bearish_total} = {bullish_bearish_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

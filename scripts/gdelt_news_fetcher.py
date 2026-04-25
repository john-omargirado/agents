import requests
import json
import calendar
from datetime import datetime
import time

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = (
    "(%22Euro+exchange+rate%22+OR+%22European+Central+Bank%22"
    "+OR+%22Eurozone+inflation%22+OR+%22Eurozone+economy%22"
    "+OR+%22Euro+forex%22+OR+%22Euro+currency+pair%22)"
)
def fetch_with_retry(url, month_label):
    """Retries monthly query until successful."""
    attempt = 1
    wait_time = 5
    while True:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.json().get('articles', [])
            else:
                print(f"  [!] {month_label}: Server Error {response.status_code}. Retrying in {wait_time}s...")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"  [!] {month_label}: Timeout/Connection Error. Attempt {attempt}. Retrying in {wait_time}s...")
        
        time.sleep(wait_time)
        wait_time = min(wait_time + 10, 60)
        attempt += 1

def run_multi_year_extraction(start_year, end_year):
    # We will store everything in this master dictionary
    master_data = {"articles": []}

    for year in range(start_year, end_year + 1):
        yearly_articles = []
        print(f"\n==== PROCESSING YEAR: {year} ====")

        for month in range(1, 13):
            # Respecting the 2023-2025 boundary logic
            last_day = calendar.monthrange(year, month)[1]
            start_str = f"{year}{month:02d}01000000"
            end_str = f"{year}{month:02d}{last_day:02d}235959"

            url = (
                f"{BASE_URL}?query={QUERY}"
                f"&mode=ArtList&format=json"
                f"&startdatetime={start_str}"
                f"&enddatetime={end_str}"
                f"&maxrecords=250"
            )

            articles = fetch_with_retry(url, f"{year}-{month:02d}")
            yearly_articles.extend(articles)
            print(f"  [OK] {year}-{month:02d}: Added {len(articles)} articles.")
            
            time.sleep(1.5) # Slightly longer delay for the long-haul run

        # Save a backup for each year just in case
        with open(f"GDELT_{year}_backup.json", "w", encoding="utf-8") as f:
            json.dump({"articles": yearly_articles}, f, indent=4)
        
        # Add to the final master list
        master_data["articles"].extend(yearly_articles)
        print(f"Year {year} complete. Total articles so far: {len(master_data['articles'])}")

    # Final save for the full 3-year span
    final_filename = f"GDELT_Final_{start_year}_{end_year}.json"
    with open(final_filename, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=4)
    
    print(f"\n✓ ALL YEARS COMPLETE.")
    print(f"Final file: {final_filename}")
    print(f"Total Database Size: {len(master_data['articles'])} articles.")

if __name__ == "__main__":
    # Runs the 3-year span as requested
    run_multi_year_extraction(2022, 2025)
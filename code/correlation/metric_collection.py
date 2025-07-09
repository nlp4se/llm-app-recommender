import argparse
import logging
import os
import time
from datetime import datetime
import pandas as pd
import requests
from google_play_scraper import app as gp_app

# Configure logging
def setup_logging(log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format
        )

def fetch_google_play_metrics(app_id):
    """Fetch metrics from Google Play Store"""
    logging.info(f"Fetching Google Play metrics for app ID: {app_id}")
    try:
        result = gp_app(app_id)
        metrics = {
            'ASO_04_rating_gp': result.get('score'),
            'ASO_07_downloads_gp': result.get('realInstalls'),
            'ASO_10_localization_gp': ','.join(result.get('languages', [])),
            'ASO_15_last_update': result.get('updated'),
            'ASO_16_system_version_gp': result.get('androidVersion'),
            'ASO_20_price_gp': result.get('price'),
        }
        logging.info(f"Successfully fetched Google Play metrics for {app_id}")
        return metrics
    except Exception as e:
        logging.error(f"Error fetching Google Play for {app_id}: {e}")
        return {k: None for k in [
            'ASO_04_rating_gp', 'ASO_07_downloads_gp', 'ASO_10_localization_gp',
            'ASO_15_last_update', 'ASO_16_system_version_gp', 'ASO_20_price_gp'
        ]}

def fetch_apple_store_metrics_itunes(app_id):
    """Fetch metrics from Apple App Store using iTunes Search API"""
    logging.info(f"Fetching Apple Store metrics for app ID: {app_id}")
    try:
        # Use iTunes Search API to get app details by ID
        url = f"https://itunes.apple.com/lookup?id={app_id}&entity=software"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['resultCount'] > 0:
            app_data = data['results'][0]
            metrics = {
                'ASO_04_rating_as': app_data.get('averageUserRating'),
                'ASO_07_downloads_as': None,  # Not available from iTunes API
                'ASO_10_localization_as': ','.join(app_data.get('languageCodesISO2A', [])),
                'ASO_15_last_update_as': app_data.get('releaseNotes'),
                'ASO_16_system_version_as': app_data.get('minimumOsVersion'),
                'ASO_20_price_as': app_data.get('price'),
            }
            logging.info(f"Successfully fetched Apple Store metrics for {app_id}")
            return metrics
        else:
            logging.warning(f"No results found for Apple Store app ID: {app_id}")
            return {k: None for k in [
                'ASO_04_rating_as', 'ASO_07_downloads_as', 'ASO_10_localization_as',
                'ASO_15_last_update_as', 'ASO_16_system_version_as', 'ASO_20_price_as'
            ]}
    except Exception as e:
        logging.error(f"Error fetching Apple Store for {app_id}: {e}")
        return {k: None for k in [
            'ASO_04_rating_as', 'ASO_07_downloads_as', 'ASO_10_localization_as',
            'ASO_15_last_update_as', 'ASO_16_system_version_as', 'ASO_20_price_as'
        ]}

def save_incremental(df, output_file, backup_interval=10):
    """Save dataframe incrementally with backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{output_file}.backup_{timestamp}"
    
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Saved progress to {output_file}")
        
        # Create backup every backup_interval saves
        if hasattr(save_incremental, 'save_count'):
            save_incremental.save_count += 1
        else:
            save_incremental.save_count = 1
            
        if save_incremental.save_count % backup_interval == 0:
            df.to_csv(backup_file, index=False)
            logging.info(f"Created backup: {backup_file}")
            
    except Exception as e:
        logging.error(f"Error saving file: {e}")

def main():
    parser = argparse.ArgumentParser(description='Collect app store metrics with logging and incremental saving')
    parser.add_argument('--input_csv', required=True, help='Input CSV file path')
    parser.add_argument('--output_csv', default='output_with_metrics.csv', help='Output CSV file path')
    parser.add_argument('--log_file', help='Log file path (optional)')
    parser.add_argument('--backup_interval', type=int, default=10, help='Backup interval (default: 10)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between API calls in seconds (default: 1.0)')
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)
    logging.info("Starting metric collection process")
    logging.info(f"Input file: {args.input_csv}")
    logging.info(f"Output file: {args.output_csv}")
    logging.info(f"Backup interval: {args.backup_interval}")
    logging.info(f"API call delay: {args.delay} seconds")

    # Check if input file exists
    if not os.path.exists(args.input_csv):
        logging.error(f"Input file not found: {args.input_csv}")
        return

    try:
        # Load input data
        logging.info("Loading input CSV file")
        df = pd.read_csv(args.input_csv, delimiter=';')
        logging.info(f"Loaded {len(df)} rows from input file")
        
        # Prepare columns
        metric_columns = [
            'ASO_04_rating_gp', 'ASO_07_downloads_gp', 'ASO_10_localization_gp',
            'ASO_15_last_update', 'ASO_16_system_version_gp', 'ASO_20_price_gp',
            'ASO_04_rating_as', 'ASO_07_downloads_as', 'ASO_10_localization_as',
            'ASO_15_last_update_as', 'ASO_16_system_version_as', 'ASO_20_price_as'
        ]
        
        for col in metric_columns:
            df[col] = None

        # Process each row
        total_rows = len(df)
        logging.info(f"Starting to process {total_rows} rows")
        
        for idx, row in df.iterrows():
            logging.info(f"Processing row {idx + 1}/{total_rows} - App: {row.get('name', 'Unknown')}")
            
            # Process Google Play metrics
            if row['google_play_id'] != '-':
                logging.info(f"Fetching Google Play data for ID: {row['google_play_id']}")
                gp_metrics = fetch_google_play_metrics(row['google_play_id'])
                for k, v in gp_metrics.items():
                    df.at[idx, k] = v
                time.sleep(args.delay)  # Rate limiting
            
            # Process Apple Store metrics using iTunes API
            if row['apple_store_id'] != '-':
                logging.info(f"Fetching Apple Store data for ID: {row['apple_store_id']}")
                as_metrics = fetch_apple_store_metrics_itunes(row['apple_store_id'])
                for k, v in as_metrics.items():
                    df.at[idx, k] = v
                time.sleep(args.delay)  # Rate limiting
            
            # Save incrementally every 5 rows
            if (idx + 1) % 5 == 0:
                save_incremental(df, args.output_csv, args.backup_interval)
                logging.info(f"Progress: {idx + 1}/{total_rows} rows processed ({(idx + 1)/total_rows*100:.1f}%)")

        # Final save
        logging.info("Processing complete. Saving final results")
        df.to_csv(args.output_csv, index=False)
        logging.info(f"Final results saved to {args.output_csv}")
        
        # Summary statistics
        processed_count = df[metric_columns].notna().any(axis=1).sum()
        logging.info(f"Summary: {processed_count}/{total_rows} rows have metric data")
        
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Saving current progress...")
        save_incremental(df, args.output_csv, args.backup_interval)
        logging.info("Progress saved. Exiting.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if 'df' in locals():
            logging.info("Attempting to save current progress...")
            save_incremental(df, args.output_csv, args.backup_interval)
        raise

if __name__ == "__main__":
    main()

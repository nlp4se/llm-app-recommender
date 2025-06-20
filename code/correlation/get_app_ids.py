import argparse
import pandas as pd
import requests
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import time
import os

def find_google_play_id(app_name: str):
    """
    Searches for a Google Play ID for the given app name by scraping the search results.
    This method is fragile and may break if Google changes its website structure.
    It takes the first result from the search page.
    """
    try:
        search_query = quote_plus(app_name)
        url = f"https://play.google.com/store/search?q={search_query}&c=apps"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # The link to the app is in an 'a' tag with a href starting with /store/apps/details
        app_link = soup.find('a', href=re.compile(r'^/store/apps/details'))

        if app_link:
            href = app_link.get('href')
            match = re.search(r'id=([a-zA-Z0-9._]+)', href)
            if match:
                return match.group(1)
    except requests.exceptions.RequestException as e:
        print(f"Error searching for '{app_name}' on Google Play: {e}")
    return None

def verify_google_play_id(app_id: str):
    """
    Verifies if a Google Play ID is valid by checking the app page URL.
    """
    if not isinstance(app_id, str) or not app_id:
        return False
    try:
        url = f"https://play.google.com/store/apps/details?id={app_id}"
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def find_apple_store_id(app_name: str):
    """
    Searches for an Apple App Store ID using the iTunes Search API.
    This is generally more reliable than scraping.
    """
    try:
        search_query = quote_plus(app_name)
        url = f"https://itunes.apple.com/search?term={search_query}&entity=software&limit=5"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['resultCount'] > 0:
            # Check for a good match in the top 5 results
            for result in data['results']:
                if app_name.lower() in result.get('trackName', '').lower():
                    return result.get('trackId')
            # If no good match, return the first result as a fallback
            return data['results'][0].get('trackId')
    except requests.exceptions.RequestException as e:
        print(f"Error searching for '{app_name}' on Apple App Store: {e}")
    except (KeyError, IndexError):
        print(f"Could not parse Apple App Store search results for '{app_name}'.")
    return None

def verify_apple_store_id(app_id):
    """
    Verifies if an Apple App Store ID is valid by checking the app page URL.
    """
    if pd.isna(app_id):
        return False
    try:
        # App store IDs are integers
        app_id_int = int(float(app_id))
        url = f"https://apps.apple.com/app/id{app_id_int}"
        response = requests.head(url, allow_redirects=True, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        # A 302 redirect is also a success for Apple store (country redirect)
        if response.status_code == 200 or response.status_code == 302:
            return True
    except (requests.exceptions.RequestException, ValueError):
        return False
    return False

def save_progress(df: pd.DataFrame, output_path: str):
    """
    Saves the current progress to the output file.
    """
    try:
        if output_path.endswith('.xlsx'):
            df.to_excel(output_path, index=False)
        elif output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        print(f"  [PROGRESS] Saved progress to '{output_path}'")
    except Exception as e:
        print(f"  [WARNING] Failed to save progress: {e}")

def process_apps(df: pd.DataFrame, output_path: str):
    """
    Processes the DataFrame to find and fill missing app IDs, saving progress after each app.
    """
    total_apps = len(df)
    
    for index, row in df.iterrows():
        app_name = row['name']
        current_app = index + 1
        print(f"\nProcessing '{app_name}' ({current_app}/{total_apps})...")

        # Handle Google Play ID
        google_id = row.get('google_play_id')
        if pd.isna(google_id) or str(google_id).strip() == '':
            print("-> Searching for Google Play ID...")
            found_id = find_google_play_id(app_name)
            if found_id and verify_google_play_id(found_id):
                print(f"  [SUCCESS] Found and verified Google Play ID: {found_id}")
                df.at[index, 'google_play_id'] = found_id
            else:
                print("  [INFO] Could not find or verify Google Play ID. Leaving blank.")
                df.at[index, 'google_play_id'] = ''
            time.sleep(1) # Be a good citizen and don't spam requests

        # Handle Apple Store ID
        apple_id = row.get('apple_store_id')
        if pd.isna(apple_id) or str(apple_id).strip() == '':
            print("-> Searching for Apple Store ID...")
            found_id = find_apple_store_id(app_name)
            if found_id and verify_apple_store_id(found_id):
                print(f"  [SUCCESS] Found and verified Apple Store ID: {found_id}")
                df.at[index, 'apple_store_id'] = found_id
            else:
                print("  [INFO] Could not find or verify Apple Store ID. Leaving blank.")
                df.at[index, 'apple_store_id'] = ''
            time.sleep(1) # Be a good citizen

        # Save progress after each app is processed
        save_progress(df, output_path)

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Fill missing app IDs in an Excel or CSV file. The script searches for apps on Google Play and Apple App Store, verifies the found IDs, and fills them in the file. Progress is saved after each app to prevent data loss."
    )
    parser.add_argument("--input-path", help="Path to the input file (Excel or CSV). It must contain a 'name' column.")
    parser.add_argument("--output-path", help="Optional path for the output file. If not provided, it will be created from input path with an '_updated' suffix.")
    parser.add_argument("--resume", action="store_true", help="Resume processing from an existing output file. If the output file exists, it will be loaded and processing will continue from where it left off.")
    
    args = parser.parse_args()
    input_path = args.input_path

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        name, ext = input_path.rsplit('.', 1)
        output_path = f"{name}_updated.{ext}"

    # Check if we should resume from existing output file
    if args.resume and os.path.exists(output_path):
        print(f"Resuming from existing output file: '{output_path}'")
        try:
            if output_path.endswith('.xlsx'):
                df = pd.read_excel(output_path)
            elif output_path.endswith('.csv'):
                df = pd.read_csv(output_path)
            print(f"Loaded {len(df)} apps from existing file.")
        except Exception as e:
            print(f"Error loading existing file: {e}")
            print("Starting fresh with input file...")
            args.resume = False
    
    # Load input file if not resuming
    if not args.resume or not os.path.exists(output_path):
        try:
            if input_path.endswith('.xlsx'):
                df = pd.read_excel(input_path)
            elif input_path.endswith('.csv'):
                df = pd.read_csv(input_path)
            else:
                print("Error: Unsupported file format. Please provide an .xlsx or .csv file.")
                return
        except FileNotFoundError:
            print(f"Error: The file '{input_path}' was not found.")
            return

    if 'name' not in df.columns:
        print("Error: Input file must have a 'name' column.")
        return
        
    print(f"Processing {len(df)} apps...")
    df = process_apps(df, output_path)

    print(f"\nFinal save to '{output_path}'...")
    save_progress(df, output_path)
    print("Done.")

if __name__ == "__main__":
    main()

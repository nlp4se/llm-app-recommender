import argparse
import pandas as pd
import requests
from urllib.parse import quote_plus
from google_play_scraper import search
import time
from difflib import SequenceMatcher

def extract_main_app_name(app_name: str):
    """
    Extracts the main app name by splitting on common separators and taking the first part.
    """
    if not app_name:
        return ""
    
    # Convert to lowercase for processing
    name = app_name.strip()
    
    # Split by common separators and take the first part (main name)
    separators = [':', '-', '–', '—', '|', '•', '·', '(', '[', '{']
    for sep in separators:
        if sep in name:
            name = name.split(sep)[0].strip()
    
    # Remove trailing punctuation
    name = name.rstrip('.,!?;:')
    
    return name

def search_google_play(keyword, k):
    """Search Google Play Store for a keyword and return top k (app_id, app_name) tuples."""
    try:
        # Use google_play_scraper to search for apps
        results = search(keyword, lang='en', country='us', n_hits=k)
        app_data = []
        for result in results:
            app_id = result.get('appId')
            app_name = result.get('title')
            if app_id and app_name:
                # Clean the app name to remove suffixes
                clean_app_name = extract_main_app_name(app_name)
                app_data.append((app_id, clean_app_name))
        return app_data
    except Exception as e:
        print(f"Error searching Google Play for '{keyword}': {e}")
        return []

def search_apple_store(keyword, k, max_retries=3):
    """Search Apple App Store for a keyword and return top k (app_id, app_name) tuples."""
    for attempt in range(max_retries):
        try:
            url = f"https://itunes.apple.com/search?term={quote_plus(keyword)}&entity=software&limit={k}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if data.get('resultCount', 0) > 0:
                for result in data.get('results', [])[:k]:
                    app_id = result.get('trackId')
                    app_name = result.get('trackName')
                    if app_id and app_name:
                        # Clean the app name to remove suffixes
                        clean_app_name = extract_main_app_name(app_name)
                        results.append((app_id, clean_app_name))
                return results
            else:
                print(f"  [RETRY {attempt + 1}/{max_retries}] Empty results for '{keyword}', retrying...")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    print(f"  [FAILED] No results found after {max_retries} attempts for '{keyword}'")
                    return []
                    
        except requests.exceptions.RequestException as e:
            print(f"  [RETRY {attempt + 1}/{max_retries}] Apple Store search failed for '{keyword}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                return []
        except Exception as e:
            print(f"  [RETRY {attempt + 1}/{max_retries}] Unexpected error for '{keyword}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
            else:
                return []
    
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, help='CSV file with keywords (header: feature)')
    parser.add_argument('--output-file', required=True, help='CSV file to save results')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top results to retrieve per store')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    keywords = df.iloc[:, 0].tolist()
    results = []
    
    for keyword in keywords:
        print(f"Searching for: {keyword}")
        
        # Get results from both stores
        try:
            gp_results = search_google_play(keyword, args.top_k)
        except Exception as e:
            print(f"Google Play search failed for '{keyword}': {e}")
            gp_results = []
            
        try:
            apple_results = search_apple_store(keyword, args.top_k)
        except Exception as e:
            print(f"Apple Store search failed for '{keyword}': {e}")
            apple_results = []
        
        # Create separate rows for each source
        # Google Play row
        gp_row = {
            "feature": keyword,
            "source": "google_play"
        }
        
        # Fill columns 1-20 with Google Play app names (now cleaned)
        for i in range(1, 21):
            if i <= len(gp_results):
                gp_row[str(i)] = gp_results[i-1][1]  # clean_app_name
            else:
                gp_row[str(i)] = ""  # Empty if less than 20 apps found
        
        results.append(gp_row)
        
        # Apple Store row
        apple_row = {
            "feature": keyword,
            "source": "apple_store"
        }
        
        # Fill columns 1-20 with Apple Store app names (now cleaned)
        for i in range(1, 21):
            if i <= len(apple_results):
                apple_row[str(i)] = apple_results[i-1][1]  # clean_app_name
            else:
                apple_row[str(i)] = ""  # Empty if less than 20 apps found
        
        results.append(apple_row)
        
        time.sleep(1)  # Be polite to the servers

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_file, index=False)
    print(f"Saved results to {args.output_file}")

if __name__ == "__main__":
    main()

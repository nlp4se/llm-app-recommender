import argparse
import pandas as pd
import requests
import re
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import time
import os
from difflib import SequenceMatcher

def extract_main_app_name(app_name: str):
    """
    Extracts the main app name by removing common suffixes, prefixes, and descriptive text.
    """
    if not app_name:
        return ""
    
    # Convert to lowercase for processing
    name = app_name.lower().strip()
    
    # Remove common prefixes
    prefixes_to_remove = ['the ', 'new ', 'best ', 'top ', 'popular ']
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    # Split by common separators and take the first part (main name)
    separators = [':', '-', '–', '—', '|', '•', '·', '(', '[', '{']
    for sep in separators:
        if sep in name:
            name = name.split(sep)[0].strip()
    
    # Remove common suffixes/descriptions
    suffixes_to_remove = [
        ' app', ' game', ' tool', ' editor', ' manager', ' viewer',
        ' chat', ' meet', ' social', ' networking', ' dating',
        ' photo', ' video', ' music', ' audio', ' camera',
        ' messenger', ' messenger app', ' chat app', ' social app',
        ' & go live', ' & meet new people', ' & chat', ' & social',
        ' - meet', ' - chat', ' - social', ' - dating'
    ]
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    
    # Remove trailing punctuation
    name = name.rstrip('.,!?;:')
    
    return name

def similarity_score(a, b):
    """
    Calculate similarity score between two strings, focusing on the main app name.
    Returns a value between 0 and 1, where 1 is identical.
    """
    if not a or not b:
        return 0
    
    # Extract main app names
    main_a = extract_main_app_name(a)
    main_b = extract_main_app_name(b)
    
    # If main names are identical, return high confidence
    if main_a == main_b and main_a:
        return 0.95
    
    # If main names are very similar, return high confidence
    if main_a and main_b:
        # Check if one is contained in the other (e.g., "meetme" vs "meetme chat")
        if main_a in main_b or main_b in main_a:
            return 0.9
        
        # Use sequence matcher for partial similarity
        sequence_similarity = SequenceMatcher(None, main_a, main_b).ratio()
        if sequence_similarity > 0.7:
            return sequence_similarity
    
    # Fallback to original full string comparison
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_google_play_id_with_verification(app_name: str):
    """
    Searches for a Google Play ID for the given app name and verifies the app name matches.
    Returns (app_id, found_app_name, confidence_score) or (None, None, 0) if not found.
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
        
        # Find the first app link
        app_link = soup.find('a', href=re.compile(r'^/store/apps/details'))
        
        if app_link:
            href = app_link.get('href')
            match = re.search(r'id=([a-zA-Z0-9._]+)', href)
            if match:
                app_id = match.group(1)
                
                # Now get the actual app name from the search result
                # Look for the app title in the search result
                app_title_element = app_link.find('div', {'class': 'vWM94c'}) or \
                                  app_link.find('div', {'class': 'DdYX5'}) or \
                                  app_link.find('div', {'class': 'WsMG1c'}) or \
                                  app_link.find('div', {'class': 'nnK0zc'})
                
                if app_title_element:
                    found_app_name = app_title_element.get_text(strip=True)
                    confidence = similarity_score(app_name, found_app_name)
                    return app_id, found_app_name, confidence
                else:
                    # If we can't find the title in search results, try to get it from the app page
                    return get_google_play_app_details(app_id, app_name)
    except requests.exceptions.RequestException as e:
        print(f"Error searching for '{app_name}' on Google Play: {e}")
    return None, None, 0

def get_google_play_app_details(app_id: str, original_app_name: str):
    """
    Gets app details from Google Play store page to verify the app name.
    """
    try:
        url = f"https://play.google.com/store/apps/details?id={app_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Updated selectors for current Google Play Store layout
        title_selectors = [
            'h1[itemprop="name"]',
            'h1.qi82n',
            'h1.Fd93Bb',
            'h1.AHFaub',
            'div[data-testid="app-title"]',
            # Add more current selectors
            'h1[data-testid="app-title"]',
            'h1[class*="title"]',
            'div[class*="title"]',
            'span[class*="title"]',
            # Look for any h1 or title-like elements
            'h1',
            'div[role="heading"]',
            'span[role="heading"]'
        ]
        
        found_app_name = None
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                found_app_name = title_element.get_text(strip=True)
                # Basic validation - make sure it's not empty and looks like an app name
                if found_app_name and len(found_app_name) > 0 and not found_app_name.isdigit():
                    break
        
        # If still no luck, try to find any text that might be the app name
        if not found_app_name:
            # Look for any text that might be the app title
            potential_titles = []
            for element in soup.find_all(['h1', 'h2', 'div', 'span']):
                text = element.get_text(strip=True)
                if text and len(text) > 2 and len(text) < 100 and not text.isdigit():
                    # Check if it contains common app name patterns
                    if any(word in text.lower() for word in ['app', 'game', 'tool', 'editor', 'manager']):
                        potential_titles.append(text)
            
            if potential_titles:
                found_app_name = potential_titles[0]  # Take the first one
        
        if found_app_name:
            confidence = similarity_score(original_app_name, found_app_name)
            return app_id, found_app_name, confidence
        else:
            # Debug: print the page structure to help identify the correct selector
            print(f"  [DEBUG] Could not find app title for {app_id}")
            print(f"  [DEBUG] Page title: {soup.title.get_text() if soup.title else 'No title'}")
            return app_id, "Unknown", 0
            
    except requests.exceptions.RequestException:
        return app_id, "Error fetching details", 0

def find_apple_store_id_with_verification(app_name: str, max_retries=3):
    """
    Searches for an Apple App Store ID and verifies the app name matches.
    Returns (app_id, found_app_name, confidence_score) or (None, None, 0) if not found.
    Includes retry mechanism for empty results.
    """
    for attempt in range(max_retries):
        try:
            search_query = quote_plus(app_name)
            url = f"https://itunes.apple.com/search?term={search_query}&entity=software&limit=5"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['resultCount'] > 0:
                best_match = None
                best_confidence = 0
                
                # Check all results and find the best match
                for result in data['results']:
                    found_name = result.get('trackName', '')
                    confidence = similarity_score(app_name, found_name)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (result.get('trackId'), found_name, confidence)
                
                if best_match:
                    return best_match
            else:
                print(f"  [RETRY {attempt + 1}/{max_retries}] Empty results for '{app_name}', retrying...")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    print(f"  [FAILED] No results found after {max_retries} attempts for '{app_name}'")
                    
        except requests.exceptions.RequestException as e:
            print(f"  [RETRY {attempt + 1}/{max_retries}] Error searching for '{app_name}' on Apple App Store: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
        except (KeyError, IndexError) as e:
            print(f"  [RETRY {attempt + 1}/{max_retries}] Could not parse Apple App Store search results for '{app_name}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
                continue
    
    return None, None, 0

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

def extract_unique_app_names(df: pd.DataFrame):
    """
    Extracts all unique app names from columns 1-20 in the app_rankings.csv format.
    """
    # Get the ranking columns (columns 1-20, which are actually columns 4-23 in 0-based indexing)
    ranking_columns = [str(i) for i in range(1, 21)]
    
    # Collect all app names from these columns
    all_app_names = set()
    for col in ranking_columns:
        if col in df.columns:
            # Get all non-null values from this column
            app_names = df[col].dropna().astype(str)
            all_app_names.update(app_names)
    
    # Convert to sorted list for consistent output
    unique_app_names = sorted(list(all_app_names))
    return unique_app_names

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

def process_apps(app_names: list, output_path: str):
    """
    Processes the list of app names to find and fill app IDs, saving progress after each app.
    Now stores clean app names (without suffixes) in the output.
    """
    # Create a DataFrame with the clean app names and additional verification columns
    clean_app_names = [extract_main_app_name(name) for name in app_names]
    
    df = pd.DataFrame({
        'original_name': app_names,  # Keep original for reference
        'clean_name': clean_app_names,  # Store clean name without suffixes
        'google_play_id': '', 
        'google_play_found_name': '',
        'google_play_confidence': 0.0,
        'apple_store_id': '',
        'apple_store_found_name': '',
        'apple_store_confidence': 0.0
    })
    total_apps = len(app_names)
    
    for index, app_name in enumerate(app_names):
        current_app = index + 1
        clean_name = clean_app_names[index]
        print(f"\nProcessing '{app_name}' -> '{clean_name}' ({current_app}/{total_apps})...")

        # Handle Google Play ID with verification
        print("-> Searching for Google Play ID...")
        app_id, found_name, confidence = find_google_play_id_with_verification(app_name)
        
        if app_id and verify_google_play_id(app_id):
            # Only accept if confidence is above threshold (0.6 = 60% similarity)
            if confidence >= 0.6:
                print(f"  [SUCCESS] Found Google Play ID: {app_id}")
                print(f"  [INFO] Found name: '{found_name}' (confidence: {confidence:.2f})")
                df.at[index, 'google_play_id'] = app_id
                df.at[index, 'google_play_found_name'] = found_name
                df.at[index, 'google_play_confidence'] = confidence
            else:
                print(f"  [WARNING] Found Google Play ID: {app_id}")
                print(f"  [WARNING] But name mismatch: '{found_name}' (confidence: {confidence:.2f})")
                print(f"  [ACTION] Marking as uncertain with '-'")
                df.at[index, 'google_play_id'] = '-'
                df.at[index, 'google_play_found_name'] = found_name
                df.at[index, 'google_play_confidence'] = confidence
        else:
            print("  [INFO] Could not find or verify Google Play ID. Leaving blank.")
            df.at[index, 'google_play_id'] = ''
            df.at[index, 'google_play_found_name'] = ''
            df.at[index, 'google_play_confidence'] = 0.0
        time.sleep(1) # Be a good citizen and don't spam requests

        # Handle Apple Store ID with verification
        print("-> Searching for Apple Store ID...")
        app_id, found_name, confidence = find_apple_store_id_with_verification(app_name)
        
        if app_id and verify_apple_store_id(app_id):
            # Only accept if confidence is above threshold (0.6 = 60% similarity)
            if confidence >= 0.6:
                print(f"  [SUCCESS] Found Apple Store ID: {app_id}")
                print(f"  [INFO] Found name: '{found_name}' (confidence: {confidence:.2f})")
                df.at[index, 'apple_store_id'] = app_id
                df.at[index, 'apple_store_found_name'] = found_name
                df.at[index, 'apple_store_confidence'] = confidence
            else:
                print(f"  [WARNING] Found Apple Store ID: {app_id}")
                print(f"  [WARNING] But name mismatch: '{found_name}' (confidence: {confidence:.2f})")
                print(f"  [ACTION] Marking as uncertain with '-'")
                df.at[index, 'apple_store_id'] = '-'
                df.at[index, 'apple_store_found_name'] = found_name
                df.at[index, 'apple_store_confidence'] = confidence
        else:
            print("  [INFO] Could not find or verify Apple Store ID. Leaving blank.")
            df.at[index, 'apple_store_id'] = ''
            df.at[index, 'apple_store_found_name'] = ''
            df.at[index, 'apple_store_confidence'] = 0.0
        time.sleep(1) # Be a good citizen

        # Save progress after each app is processed
        save_progress(df, output_path)

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Extract unique app names from app_rankings.csv and find their Google Play and Apple App Store IDs. The script searches for apps on both platforms, verifies the found IDs, and saves the results to a CSV file. Progress is saved after each app to prevent data loss."
    )
    parser.add_argument("--input-path", required=True, help="Path to the app_rankings.csv file.")
    parser.add_argument("--output-path", help="Optional path for the output file. If not provided, it will be created from input path with an '_app_ids' suffix.")
    parser.add_argument("--resume", action="store_true", help="Resume processing from an existing output file. If the output file exists, it will be loaded and processing will continue from where it left off.")
    
    args = parser.parse_args()
    input_path = args.input_path

    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        name, ext = input_path.rsplit('.', 1)
        output_path = f"{name}_app_ids.{ext}"

    # Check if we should resume from existing output file
    if args.resume and os.path.exists(output_path):
        print(f"Resuming from existing output file: '{output_path}'")
        try:
            if output_path.endswith('.xlsx'):
                df = pd.read_excel(output_path)
            elif output_path.endswith('.csv'):
                df = pd.read_csv(output_path)
            print(f"Loaded {len(df)} apps from existing file.")
            # Get the list of app names from the existing file
            app_names = df['original_name'].tolist()
        except Exception as e:
            print(f"Error loading existing file: {e}")
            print("Starting fresh with input file...")
            args.resume = False
    
    # Load input file and extract app names if not resuming
    if not args.resume or not os.path.exists(output_path):
        try:
            if input_path.endswith('.xlsx'):
                df_input = pd.read_excel(input_path)
            elif input_path.endswith('.csv'):
                df_input = pd.read_csv(input_path)
            else:
                print("Error: Unsupported file format. Please provide an .xlsx or .csv file.")
                return
        except FileNotFoundError:
            print(f"Error: The file '{input_path}' was not found.")
            return

        # Extract unique app names from the ranking columns
        print("Extracting unique app names from ranking columns...")
        app_names = extract_unique_app_names(df_input)
        print(f"Found {len(app_names)} unique app names.")

    print(f"Processing {len(app_names)} apps...")
    df = process_apps(app_names, output_path)

    print(f"\nFinal save to '{output_path}'...")
    save_progress(df, output_path)
    print("Done.")

if __name__ == "__main__":
    main()

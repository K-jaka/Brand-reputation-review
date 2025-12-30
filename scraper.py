import requests
import pandas as pd
from datetime import datetime
import time

def scrape_brand_data():
    api_url = "https://web-scraping.dev/api/graphql"
    
    gql_query = """
    query GetReviews($first: Int, $after: String) {
      reviews(first: $first, after: $after) {
        edges {
          node {
            rid
            text
            rating
            date
          }
          cursor
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
    """
    
    reviews_list = []
    has_next = True
    after_cursor = None
    page_count = 0

    print("Extracting review data...")

    while has_next:
        page_count += 1
        payload = {
            "query": gql_query,
            "variables": {"first": 20, "after": after_cursor}
        }
        
        try:
            response = requests.post(api_url, json=payload, timeout=15)
            data = response.json()
            edges = data['data']['reviews']['edges']
            
            if not edges:
                break

            for edge in edges:
                node = edge['node']
                # real string to datetime
                real_date = pd.to_datetime(node['date'])
                
                # keep only if == 2023
                if real_date.year == 2023:
                    reviews_list.append({
                        "date": real_date,
                        "title": f"Review {node['rid']}",
                        "text": node['text'],
                        "rating": node['rating']
                    })
                elif real_date.year < 2023:
                    # if == 2022 then stop
                    print(f"Reached date {real_date.date()}. Stopping scrape.")
                    has_next = False
                    break
            
            # pagination if there are still 2023 dates
            if has_next:
                has_next = data['data']['reviews']['pageInfo']['hasNextPage']
                after_cursor = data['data']['reviews']['pageInfo']['endCursor']
                print(f"Page {page_count} processed. Total 2023 reviews found: {len(reviews_list)}")
                time.sleep(0.3)
            
        except Exception as e:
            print(f"Connection Error: {e}")
            break

    # save data
    if reviews_list:
        df = pd.DataFrame(reviews_list)
        df = df.sort_values(by='date')
        df.to_csv("scraped_reviews.csv", index=False)
        print(f"\nFiltered to 2023 and saved {len(df)} authentic 2023 reviews.")
        print(f"Date Range in File: {df['date'].min()} to {df['date'].max()}")
    else:
        print("No reviews from 2023 found on the site.")

if __name__ == "__main__":
    scrape_brand_data()
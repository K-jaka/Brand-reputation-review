import requests
import pandas as pd
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
            # FIX: raise immediately on HTTP errors (4xx, 5xx) instead of
            # crashing later with a confusing KeyError on data['data']
            response.raise_for_status()
            data = response.json()

            # FIX: check for GraphQL-level errors (separate from HTTP errors)
            if "errors" in data:
                print(f"GraphQL error: {data['errors']}")
                break

            edges = data['data']['reviews']['edges']

            if not edges:
                break

            # FIX: use a flag instead of setting has_next inside the loop
            # to avoid the outer pagination block running after the break
            stop_scraping = False
            for edge in edges:
                node = edge['node']
                real_date = pd.to_datetime(node['date'])

                if real_date.year == 2023:
                    reviews_list.append({
                        "date": real_date,
                        "title": f"Review {node['rid']}",
                        "text": node['text'],
                        "rating": node['rating']
                    })
                elif real_date.year < 2023:
                    print(f"Reached date {real_date.date()}. Stopping scrape.")
                    stop_scraping = True
                    break

            if stop_scraping:
                break

            # paginate if more pages exist
            has_next = data['data']['reviews']['pageInfo']['hasNextPage']
            after_cursor = data['data']['reviews']['pageInfo']['endCursor']
            print(f"Page {page_count} processed. Total 2023 reviews found: {len(reviews_list)}")
            time.sleep(0.3)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error on page {page_count}: {e}")
            break
        except requests.exceptions.ConnectionError:
            print(f"Connection failed on page {page_count}. Check your internet connection.")
            break
        except requests.exceptions.Timeout:
            print(f"Request timed out on page {page_count}. Try increasing timeout.")
            break
        except Exception as e:
            print(f"Unexpected error on page {page_count}: {e}")
            break

    # save data
    if reviews_list:
        df = pd.DataFrame(reviews_list)
        df = df.sort_values(by='date')
        df.to_csv("scraped_reviews.csv", index=False)
        print(f"\nSaved {len(df)} reviews from 2023.")
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    else:
        print("No reviews from 2023 found.")


if __name__ == "__main__":
    scrape_brand_data()
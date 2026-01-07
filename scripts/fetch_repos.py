import requests


# --- Configuration ---
GITHUB_API_URL = "https://api.github.com/search/repositories"
SEARCH_QUERY1 = 'language:Python flask stars:10..500'
SEARCH_QUERY2 = 'language:Java stars:10..500'
SEARCH_QUERY3 = 'language:Python django stars:10..500'
SEARCH_QUERY4 = 'language:JavaScript express stars:10..500'
SEARCH_QUERY5 = 'language:TypeScript api stars:10..500'

SEARCH_QUERY6 = 'filename:Dockerfile stars:10..500' # didn't work
SEARCH_QUERY7 = 'filename:terraform.tf stars:10..500' # didn't work
SEARCH_QUERY8 = 'topic:kubernetes stars:10..500'
SEARCH_QUERY9 = 'topic:ansible stars:10..500'
SEARCH_QUERY10 = 'language:Go devops stars:10..500'

SEARCH_QUERY11 = 'extension:ipynb stars:10..500' # didn't work
SEARCH_QUERY12 = 'topic:mlflow stars:10..500'
SEARCH_QUERY13 = 'topic:huggingface stars:10..500'
SEARCH_QUERY14 = 'topic:data-pipeline stars:10..500'

SEARCH_QUERY15 = 'topic:cli stars:10..500'
SEARCH_QUERY16 = 'language:Go cli stars:10..500'
SEARCH_QUERY17 = 'language:Rust tool stars:10..500'
SEARCH_QUERY18 = 'language:Python argparse stars:10..500'

SEARCH_QUERY19 = 'topic:security-testing stars:10..500'
SEARCH_QUERY20 = 'topic:auth stars:10..500'
SEARCH_QUERY21 = 'topic:penetration-testing stars:10..500'

# Replacement for query 6, 7 and 11
SEARCH_QUERYa = 'topic:terraform stars:10..500'
SEARCH_QUERYb = 'topic:docker stars:10..500'
SEARCH_QUERYc = 'language:"Jupyter Notebook"  stars:10..500'

RESULTS_PER_PAGE = 50
PAGE_NUMBER = 1

GITHUB_TOKEN = "gh**********Pc" # My Github API token
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}"
}


def fetch_github_repos(query, per_page, page):
    params = {
        'q': query,
        'per_page': per_page,
        'page': page,
        'sort': 'updated',  # Sort by recently updated
        'order': 'desc'
    }

    print(f"-> Searching GitHub for: '{query}' (Page {page})")

    try:
        response = requests.get(GITHUB_API_URL, params=params, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error occurred: {err}")
        if response.status_code == 403:
            print("Rate limit likely exceeded.")
        return None
    except requests.exceptions.RequestException as err:
        print(f"An error occurred during the request: {err}")
        return None


def extract_repo_urls(search_data):
    if not search_data or 'items' not in search_data:
        return []

    repos = []
    for item in search_data['items']:
        repo_info = {
            'full_name': item.get('full_name'),
            'url': item.get('html_url')
        }
        repos.append(repo_info)
    return repos


# --- Main Execution ---
if __name__ == "__main__":
    search_results1 = fetch_github_repos(SEARCH_QUERY1, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results2 = fetch_github_repos(SEARCH_QUERY2, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results3 = fetch_github_repos(SEARCH_QUERY3, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results4 = fetch_github_repos(SEARCH_QUERY4, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results5 = fetch_github_repos(SEARCH_QUERY5, RESULTS_PER_PAGE, PAGE_NUMBER)
    #search_results6 = fetch_github_repos(SEARCH_QUERY6, RESULTS_PER_PAGE, PAGE_NUMBER) # didn't work
    #search_results7 = fetch_github_repos(SEARCH_QUERY7, RESULTS_PER_PAGE, PAGE_NUMBER) # didn't work
    search_results8 = fetch_github_repos(SEARCH_QUERY8, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results9 = fetch_github_repos(SEARCH_QUERY9, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results10 = fetch_github_repos(SEARCH_QUERY10, RESULTS_PER_PAGE, PAGE_NUMBER)
    #search_results11 = fetch_github_repos(SEARCH_QUERY11, RESULTS_PER_PAGE, PAGE_NUMBER) # didn't work
    search_results12 = fetch_github_repos(SEARCH_QUERY12, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results13 = fetch_github_repos(SEARCH_QUERY13, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results14 = fetch_github_repos(SEARCH_QUERY14, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results15 = fetch_github_repos(SEARCH_QUERY15, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results16 = fetch_github_repos(SEARCH_QUERY16, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results17 = fetch_github_repos(SEARCH_QUERY17, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results18 = fetch_github_repos(SEARCH_QUERY18, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results19 = fetch_github_repos(SEARCH_QUERY19, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results20 = fetch_github_repos(SEARCH_QUERY20, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_results21 = fetch_github_repos(SEARCH_QUERY21, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_resultsa = fetch_github_repos(SEARCH_QUERYa, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_resultsb = fetch_github_repos(SEARCH_QUERYb, RESULTS_PER_PAGE, PAGE_NUMBER)
    search_resultsc = fetch_github_repos(SEARCH_QUERYc, RESULTS_PER_PAGE, PAGE_NUMBER)

    search_result_all = [search_resultsa, search_resultsb, search_resultsc] # add rest of the queries in search_result_all as well

    for search_results in search_result_all:
        if search_results:
            repo_list = extract_repo_urls(search_results)

            print(f"\n--- GitHub Repository URLs ---")
            if repo_list:
                for i, repo in enumerate(repo_list, 1):
                    print(f"{repo['url']}")
            else:
                print("No repositories found for the given query.")

            print(f"\nTotal Repositories Found (on this page): {len(repo_list)}")
            print(f"Total Matches on GitHub: {search_results.get('total_count', 'Unknown')}")
        else:
            print("\nFailed to retrieve search results.")

# Later I copied all the resultant repositories in repos.txt
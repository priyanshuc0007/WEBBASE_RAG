import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator=' ', strip=True)
    return text

def extract_all_links(soup, base_url):
    links = []
    for anchor in soup.find_all('a', href=True):
        link = anchor['href']
        full_link = urljoin(base_url, link)
        if full_link not in links:
            links.append(full_link)
    return links

def scrape_website(base_url):
    # Scrape the main page
    main_page_text = scrape_page(base_url)
    print(f"Scraped text from the main page ({base_url})")

    # Parse the main page to find all links
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = extract_all_links(soup, base_url)

    # Scrape each linked page
    all_text_data = main_page_text + "\n\n"
    for link in links:
        try:
            page_text = scrape_page(link)
            all_text_data += page_text + "\n\n"
            print(f"Scraped text from linked page ({link})")
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")

    return all_text_data

# Main URL of the website
url = "https://relinns.com/"
all_text_data = scrape_website(url)

# Save the data to a text file or further process it as needed
with open('data/all_text_data.txt', 'w', encoding='utf-8') as f:
    f.write(all_text_data)

print("All text data has been scraped and saved.")

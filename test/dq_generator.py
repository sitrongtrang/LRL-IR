from datasets import load_dataset
import re
import requests
from bs4 import BeautifulSoup
import json
import os

dataset = load_dataset("wikimedia/wikipedia", "20231101.km")["train"]

file_names = os.listdir('dq')
ids = [int(re.match(r'(\d+)\.json', f).group(1)) for f in file_names if re.match(r'(\d+)\.json', f)]

def get_first_sentence(text):
    sentences = re.split(r'(?<=\.)\s+', text)
    return sentences[0] if sentences else text

def get_wikipedia_links(page_title, language_code="en"):
    url = f"https://{language_code}.wikipedia.org/wiki/{page_title}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to retrieve page: {url}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/wiki/') and ':' not in href[6:]:
            full_url = f"https://{language_code}.wikipedia.org{href}"
            links.append(full_url)
    
    return links

for article in dataset:
    id = int(article['id'])
    if id in ids: continue
    print(id)
    
    url = article['url']
    title = article['title']
    text = article['text']
    
    # Extract first sentence and add to dictionary
    first_sentence = get_first_sentence(text)
    query = re.sub(rf'\b{re.escape(title)}\b', "", first_sentence)
    wiki_links = get_wikipedia_links(title, "km")
    non_relevence = []
    
    for article_0 in dataset:
        url_0 = article_0['url']
        if url_0 != url and url_0 not in wiki_links:
            non_relevence.append(url_0)
    
    art_save = {"id" : id, 
                    "url" : url, 
                    "title" : title, 
                    "text" : text, 
                    "query": query, 
                    "slightly_relevence" : wiki_links,
                    "non_relevence" : non_relevence
                    }
    
    with open('dq/' + str(id) + '.json', 'w', encoding='utf-8') as f:
        json.dump(art_save, f, indent=2, ensure_ascii=False)
    

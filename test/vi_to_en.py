from bs4 import BeautifulSoup
import requests
import pandas as pd

def get_english_wikipedia_url(vi_url):
    response = requests.get(vi_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    language_links = soup.find('div', {'id': 'p-lang-btn'})
    if language_links:
        for link in language_links.find_all('a'):
            lang = link.get('lang')
            if lang == 'en':
                return link.get('href')
    return ""

ori_df = pd.read_csv('output.csv')
df = ori_df.copy()
df = df.drop(columns=['text'])
df['en_url'] = df['url'].apply(get_english_wikipedia_url)
# df.loc[0, 'en_url'] = get_english_wikipedia_url(df.loc[0, 'url'])
df.to_csv('output_en.csv', index=False)

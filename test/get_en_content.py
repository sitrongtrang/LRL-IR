import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_wiki_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = soup.find('div', {'class': 'mw-parser-output'})
            
            elements = content.find_all(['p', 'dd'])
            text = ' '.join([elem.get_text() for elem in elements])
            return text
        else:
            return None
    except:
        return None

ori_df = pd.read_csv('2002_en.csv')
df = ori_df.copy()
df = df.drop(columns=['url'])
df['text'] = df['en_url'].apply(get_wiki_text)
df.to_csv('2002_en_text.csv', index=False)

import os
import json
import re
import requests
from bs4 import BeautifulSoup

source = "tradition"
destination = "a"
error = "errors.txt"
reasons = "reasons.txt"

def get_first_paragraph(page_id, page_title, language_code="en"):
    url = f"https://{language_code}.wikipedia.org/wiki/{page_title}"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        new_url = f"https://{language_code}.wikipedia.org/wiki/?oldid={page_id}"
        response = requests.get(new_url)
        if response.status_code != 200:
            print(f"Failed to retrieve page: {url}")
            return "Failed to retrieve page"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    paragraphs = soup.find(id='bodyContent').find_all('p')
    
    for i in range(len(paragraphs)):
        if paragraphs[i].get_text(strip=True) == "":
            continue
        else:
            return paragraphs[i].get_text(strip=True)
    
    return ""
    
def get_first_sentence(text):
    sentence_endings = r'[។៕៖!?\n]'
    sentences = re.split(f"({sentence_endings})", text)
    if len(sentences) > 1:
        return (sentences[0] + sentences[1])
    
    elif len(sentences) == 1:
        return sentences[0]

    return ""
    
def add_to_error(error, file):
    with open(file, 'r+') as f:
        content : set = eval(f.read())
        content.add(error)
        
        f.seek(0)
        f.write(repr(content))
        
def add_to_log(log, file):
    with open(file, 'a', encoding='utf-8') as f:
        f.write(log)
        
def remove_a_from_b(b, a):
    b = (re.sub(rf'{re.escape(a)}', "", b))
    return b

curr = 0
for filename in os.listdir(source):
    curr += 1
    if filename in os.listdir(destination):
        continue
    
    file_path = os.path.join(source, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        id = data["id"]
        title = data["title"]
        text = data["text"]
        
        url = f"https://km.wikipedia.org/wiki/{title}"
        
        first_paragraph = get_first_paragraph(id, title, "km")
        if first_paragraph == "Failed to retrieve page":
            add_to_error(filename, error)
            add_to_log("Failed to retrieve page: " + url + " " + filename + "\n", reasons)
            continue
        elif first_paragraph == "":
            add_to_error(filename, error)
            add_to_log("Failed to get first paragraph: " + url + " " + filename + "\n", reasons)
            continue
        
        first_sentence = get_first_sentence(first_paragraph)
        if first_sentence == "":
            add_to_error(filename, error)
            add_to_log("Failed to generate query: " + url + " " + filename + "\n", reasons)
            continue
        
        query = remove_a_from_b(first_sentence, title)
        
        new_text = remove_a_from_b(text, first_sentence)
        
        if new_text == "":
            query = title
            new_text = remove_a_from_b(text, query)

        # Remove query from content
        data["text"] = new_text
        
        # Remove title from query
        data["query"] = query
        
    file_path = os.path.join(destination, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(str(curr) + " " + filename + " " + query)

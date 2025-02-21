import csv
import os
import xml.etree.ElementTree as ET

import pandas as pd

def merge_sentences(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for p in root.findall(".//p"):
        sentences = []
        first_s_id = None

        for i, s in enumerate(p.findall("s")):
            if i == 0:
                first_s_id = s.get("id")  
            sentences.append(s.text.strip())

        if sentences:
            new_s = ET.Element("s", id=first_s_id)
            new_s.text = " ".join(sentences)

            for s in p.findall("s"):
                p.remove(s)
            p.append(new_s)

    return tree 

def extract_sentence(sentences, xml_path):
    tree = merge_sentences(xml_path)
    root = tree.getroot()
    sentences += [s.text.strip() for s in root.findall(".//s") if s.text]


en_folder = "en/TED2020/raw/en"
vi_folder = "vi/TED2020/raw/vi"

csv_file = "test_bitext.csv"
df = pd.read_csv(csv_file)

i = 0
en_sentences = []
vi_sentences = []

for filename in os.listdir(en_folder):
    if i > 256:
        break
    i += 1
    if filename.startswith("ted2020-") and filename.endswith(".xml") and os.path.exists(os.path.join(vi_folder, filename)):
        en_xml_path = os.path.join(en_folder, filename)
        vi_xml_path = os.path.join(vi_folder, filename)

        extract_sentence(en_sentences, en_xml_path)
        extract_sentence(vi_sentences, vi_xml_path)
        print(filename, len(en_sentences), len(vi_sentences))


num_existing_rows = len(df)
num_sentences = len(en_sentences)
df.iloc[:num_existing_rows - 1, df.columns.get_loc("source")] = en_sentences[:num_existing_rows]
df.iloc[:num_existing_rows - 1, df.columns.get_loc("target")] = vi_sentences[:num_existing_rows]

if num_sentences > num_existing_rows:
    extra_en_sentences = en_sentences[num_existing_rows:]
    extra_vi_sentences = vi_sentences[num_existing_rows:]
    extra_rows = pd.DataFrame({"source": extra_en_sentences, "target": extra_vi_sentences})
    df = pd.concat([df, extra_rows], ignore_index=True)

# Save the updated CSV
df.to_csv(csv_file, index=False)


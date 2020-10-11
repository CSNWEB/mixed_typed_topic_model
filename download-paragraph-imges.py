import json
import urllib.request
import os
from tqdm import tqdm

failed = []

with open('../data/paragraphs_v1.json', mode='r') as json_file:
    data = json.load(json_file)
    line_count = 0
    for row in tqdm(data):
        print(f'\timage id {row["image_id"]} url: {row["url"]}.')
        file_name = f'../data/paragraphs/images/{row["image_id"]}.jpg'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        if os.path.exists(file_name):
            print(file_name + " already exists")
            continue
        print("Downloading: " + row["url"] + " to: " + file_name)
        try:
            urllib.request.urlretrieve(row["url"], file_name)
        except Exception:
            print("Failed to download: " + row["url"])
            failed.append(row)
            continue
        line_count += 1
    print(f'Processed {line_count} lines.')

with open("failed-paragraphs.txt", "w") as f:
    for s in failed:
        f.write(str(s))

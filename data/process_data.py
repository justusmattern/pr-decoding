import json
from operator import itemgetter

with open('data.json', 'r') as f:
    data = json.load(f)


movies = dict()
for obj in data:
    movies[obj['user']] = 0

for obj in data:
    movies[obj['user']] += 1

res = dict(sorted(movies.items(), key = itemgetter(1), reverse = True)[:10])

all_texts = []
for author in res.keys():
    reviews = []
    for obj in data:
        if obj['user'] == author:
            reviews.append(obj['review'])
    
    all_texts.extend(reviews)
        
    with open(f'{author}.txt', 'w') as f:
        for t in reviews:
            f.write(t.replace('\n', ' ')+'\n')

with open('all.txt', 'w') as f:
    for t in all_texts:
        f.write(t.replace('\n', ' ')+'\n')


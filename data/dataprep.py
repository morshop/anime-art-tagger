import json
import pandas as pd
from pathlib import Path
import shutil
import os

# extracting img-tag data from full metadata

tags_dict = {}
for file in Path('metadata').rglob('*.json'):
    tags_dict.update({
        item['id']: [
            tag['name']
            for tag
            in item['tags']
            if (
                tag['name'] not in ['1girl', 'solo', 'breasts'] # no need to state the obvious
                and tag['category'] == '0'
            )
        ]
        for item
        in [
            json.loads(line)
            for line
            in open(file, 'r', encoding='utf-8')
        ] if (
            10 <= int(item['created_at'][2:4]) <= 20
            and item['rating'] == 's'
            and (
                {'1girl', 'solo'}.issubset(
                    {tag['name'] for tag in item['tags']}
                )
            )
            and not (
                bool(int(item['parent_id']))
                or item['is_deleted']
                or item['is_banned']
            )
        )
    })


# keeping the most common tags

vc = pd.Series(
    [tag for tags in tags_dict.values() for tag in tags]
).value_counts()

top_tags = pd.Series(vc[vc > 100000].index.rename('tag'))

top_tags.to_csv('top_tags.csv', index=False)

# vc>100 gives 6394 tags
# vc>1000 gives 2177 tags
# vc>10000 gives 483 tags
# vc>100000 gives 56 tags


raw_imgs = [f for f in Path('all-images').rglob('*.jpg')]
img_ids = {f.stem for f in raw_imgs}


# filtering img-tags data

df = pd.DataFrame(
    {
     'id':tags_dict.keys(),
     'tags':tags_dict.values()
    }
)

df.tags = df.tags.apply(
    lambda l: list(
        set(top_tags).intersection(set(l))
    )
)

img_tags = df[df.tags.apply(len).gt(10) & df.id.isin(img_ids)]

img_tags.to_csv('img_tags.csv', index=False)


# img extraction

filt_ids = img_ids.intersection(set(img_tags.id))
filt_imgs = [f for f in raw_imgs if f.stem in filt_ids]

filt_path = Path('filtered-images')

for stage in ['train', 'val', 'test']:
    os.makedirs(filt_path / stage, exist_ok=True)

for idx, img in enumerate(filt_imgs):
    stage = (('train','val')[not idx%5], 'test')[not idx%10]
    dest = filt_path / stage / img.name
    shutil.copy(img, dest)

# in case 80K images are too much

# less_path = Path('less-images')

# for stage in ['train', 'val', 'test']:
#     curr_imgs = [f for f in (filt_path / stage).rglob('*.jpg')]
#     less_imgs = [f for i, f in enumerate(curr_imgs) if not i%10]
    
#     os.makedirs(less_path / stage, exist_ok=True)

#     for img in less_imgs:
#         dest = less_path / stage / img.name
#         shutil.move(img, dest)
## changelog:
### 2022/09/20
- pytorch and torchvision version updates
- model updated from EfficientNetB4 to EfficientNetV2S
- overall results improved (50.6% -> 52.1% F1)
- training time was shortened by a whole lot (10 hr diff)

------

# anime art tagger

POC for a multi-label classifier for anime-style art,  
using a neural network with an EfficientNetV2S base.

## data

the dataset consists of images and their corresponding metadata as posted on the anime image board danbooru (no link because it's full of lewds), scraped by gwern to form the [Great Danbooru Dataset™](https://www.gwern.net/Danbooru2021).

for this initial version, i used the ["safe" subset of the 2020 release](https://www.kaggle.com/datasets/muoncollider/danbooru2020), then narrowed it down to only include: 

- images from 2010 and beyond.
- images tagged with both `1girl` and `solo`.
- images with an empty `parent id` to avoid duplicates.
- images **not** marked as `banned` or `deleted`.
- tags under the `general` category.
- tags that appear over 100K times in the subset,  
excluding `1girl`, `solo`, and `breasts`, making 56 in total.
- images with more than 10 tags after previous cuts.

the full code can be seen in `data/dataprep.py`.

## model

the model has an EfficientNetV2S base, on top of which i added layers and functions from previous work by [RF5](https://github.com/RF5/danbooru-pretrained) and [anthony](https://github.com/anthony-dipofi/danbooru-tagger).

training was done in colab, which is a garbage platform. this means:
- i had to cut the *filtered* dataset by 90% so it could be fetched from the drive.
- the code in `model/train_nn.ipynb` had some additional snippets that aren't in the file.

the model was trained for 146 epochs (\~7.5hrs) before it reached an early stop condition.  
testing was done locally and is presented in `model/test_nn.ipynb`.

## results

the model yielded an F1 score of 52.1% for a probability threshold of 35%, which is alright for such an early stage.  
`showcase.ipynb` presents the current results, including unsupervised inference for 4 pics.

## plans

the whole thing is still in its infancy and will probably take quite a while to finish, but it's fun to do so i'm gonna keep working on it in my spare time.

future plans include, among others:
- implementing pytorch lightning (currently underway)
- experiment management and EDA
- ditching colab for AWS
- connecting to the DB directly instead of using kaggle

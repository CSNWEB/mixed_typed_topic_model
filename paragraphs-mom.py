from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from PIL import Image
import numpy as np
from tqdm import tqdm
import mom
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import torch
import spacy
from scipy.stats import multivariate_normal
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), ])

preprocessed = pickle.load(open('preprocessed-paragraphs.pkl', 'rb'))
keys = list(preprocessed.keys())
keys.sort(key=int)
images = np.zeros((len(keys), 512))
i = 0
for key in keys:
    images[i] = torch.squeeze(preprocessed[key])
    i += 1

seen = set()
uniq = []
texts = []
i = 0

NLP = spacy.load("en_core_web_md")
images = np.zeros((len(keys), 512))
with open('../data/paragraphs_v1.json', mode='r') as json_file:
    data = json.load(json_file)
    for row in tqdm(data):
        if row['image_id'] not in seen:
            uniq.append(row['image_id'])
            seen.add(row['image_id'])
            if row['image_id'] in keys:
                parsed_doc = NLP(row['paragraph'])
                text = []
                for token in parsed_doc:
                    if not token.is_stop:
                        text.append(token.lemma_)
                texts.append(' '.join(text))
                images[i] = torch.squeeze(preprocessed[row['image_id']])
                i += 1
        else:
            print(row)

vectorizer = CountVectorizer(min_df=10, max_df=0.5, dtype=np.float32)
bows = vectorizer.fit_transform(texts)
bows_matrix = bows.toarray()
print("Character-document matrix:")
print(bows_matrix.shape)
bows_matrix = bows_matrix[bows_matrix.sum(axis=1) > 2]
print("Character-document matrix:")
print(bows_matrix.shape)
images = images
mixed = np.concatenate((images, bows_matrix), axis=1)

# Image only
# Number  of pixels
c = images.shape[1]
# Number of topics
k = 10
image_mu, image_alpha, image_sigmas, *_ = mom.fit(images, c, 1, k)

# Text only
text_mu, text_alpha, *_ = mom.fit(bows_matrix, 0, 1, k)

# Mixed
mu, alpha, sigmas, *_ = mom.fit(mixed, c, 1, k)

sigmas[sigmas < 0] *= -1
# %%


def get_top_images(images, uniq, mu):
    top_images = []
    for i in range(k):
        probabilities = multivariate_normal.pdf(images, mean=mu[:, i], cov=1)
        top = np.argsort(probabilities)[::-1][:4]
        for j in range(4):
            with Image.open('../data/paragraphs/images/' + str(uniq[top[j]]) + '.jpg') as img:
                top_images.append(transform(img))
    return top_images


feature_names = vectorizer.get_feature_names()
# %%
top = 10
topics = []
# comment out below for mixed
# text_mu = mu[c:]
# text_alpha = alpha
for i in tqdm(range(k)):
    # most_likely = np.argsort(mu[c:, i])[::-1][:top]
    most_likely = np.argsort(text_mu[:, i])[::-1][:top]
    # topic = [i, alpha[i]]
    topic = [i, text_alpha[i]]
    for j in range(top):
        topic.append(feature_names[most_likely[j]])
    topics.append(topic)

text_topics_df = pd.DataFrame(topics)
text_topics_df.to_csv("../data/paragraphs/text3_k10_spacy.csv")


top_images = get_top_images(images, uniq, mu[:c])
# %%
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

fig = plt.figure(figsize=(11, 7.5), dpi=200)
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(k//2, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )


for i, (ax, im) in enumerate(zip(grid, top_images[20:])):
        # Iterating over the grid returns the Axes.
    # ax.axis('off')
    if i == 0:
        ax.set_title('highest prob.')
        ax.annotate("Topic", xy=(0, 0), xytext=(15, 54),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    if i == 1:
        ax.set_title('$2^{nd}$ highest prob.')
    if i == 2:
        ax.set_title('$3^{rd}$ highest prob.')
    if i == 3:
        ax.set_title('$4^{th}$ highest prob.')
    # if i == 4:
    #     ax.set_title('5th highest prob.')

    if i % 4 == 0:
        ax.set_ylabel(i//4+6, rotation=0, size='large', labelpad=15)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.axis('off')
    ax.imshow(im)
fig.tight_layout()
plt.annotate("Topic", xy=(1, 1))
plt.savefig('last_5_image_topics.pgf', bbox_inches='tight', pad_inches=0)
# plt.show()

# %%

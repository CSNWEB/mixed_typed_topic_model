# Mixed type topic models (for image and text) learned with tensor decomposition
Code used for my master thesis "Method of Moments for Mixed-Domain Topic Models"

To run this code first install all dependencies in requirements.txt. The real world dataset can be found here: https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html.
 Save the paragraphs_v1.json file in a separate data folder (its searched for in ../data). Afterwards you can download the images using `download-praragraph-images.py`. The prepocessing with the first 17 layers of a pretrained Resnet18 is done by `preprocess-paragraphs.py`


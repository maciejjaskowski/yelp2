import lasagne
import vgg16
import pickle
from scipy import misc
import pandas as pd
import os
import numpy as np
import sys
import getopt

import io
import skimage.transform

from lasagne.utils import floatX

default_vgg16_path = '/Users/maciej/Projects/datascience/Lasagne-Recipes/modelzoo/vgg16.pkl'
photos_dir_path = '../yelp-data/train_photos/'
output_path = '../yelp-data/train-photos2.h5'

optlist, args = getopt.getopt(sys.argv[1:], '', [
    'vgg16_path=',
    'fraction=',
])

d = {'vgg16_path': default_vgg16_path,
     'fraction': 0.00001}

for o, a in optlist:
    if o in ("--vgg16_path",):
        d['vgg16_path'] = a
    if o in ("--fraction",):
        d['fraction'] = a

network = vgg16.build_model()
fc7 = network['fc7']
output_layer = network['fc8']

vgg16_weights = pickle.load(open(d['vgg16_path']))
param_values = vgg16_weights['param values']
MEAN_IMAGE = vgg16_weights['mean value']
CLASSES = vgg16_weights['synset words']
lasagne.layers.set_all_param_values(output_layer, param_values)

# import urllib
# index = urllib.urlopen('http://www.image-net.org/challenges/LSVRC/2012/ori_urls/indexval.html').read()
# image_urls = index.split('<br>')
#
# np.random.seed(23)
# np.random.shuffle(image_urls)
# image_urls = image_urls[:2]


def resize_and_crop(im):
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    return im[h//2-112:h//2+112, w//2-112:w//2+112]


def prep_for_network(im):

    rawim = np.copy(im).astype('uint8')
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE.reshape(3, 1, 1)
    return rawim, floatX(im[np.newaxis])


#def fetch_img(url):
#    ext = url.split('.')[-1]
#    return plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)


def prep_y(train):
    n_labels = 9
    # remove NaNs
    train2 = train[train['labels'].notnull()]

    train3 = train2
    train3['labels_list'] = train2['labels'].str.split().values.tolist()

    for i in range(n_labels):
      train3[str(i)] = train3['labels_list'].apply(lambda row: str(i) in row).astype(int)

    return train3.iloc[:, -9:]


FC7_COLS = ['fc7_' + str(i) for i in range(4096)]


processed = 0

def to_df(photo_to_biz, train_y, paths):
    global processed
    try:
        imgs = []
        for path in paths:
            photo_id = int(path.split('/')[-1][:-4])
            business_id = photo_to_biz.loc[photo_id, 'business_id']
            rawim, im = prep_for_network(resize_and_crop(misc.imread(path)))
            if business_id not in [430, 1627, 2661, 2941]:
                imgs.append((photo_id, business_id, rawim, im))
    except IOError:
        print('bad path: ' + path)

    imgs_only = np.concatenate([i[-1] for i in imgs], axis=0)
    fc7_outs = list(np.array(lasagne.layers.get_output(fc7, imgs_only, deterministic=True).eval()))

    photo_ids, business_ids = zip(*[[photo_id, business_id] for (photo_id, business_id, _, _) in imgs])
    rows = pd.concat([pd.DataFrame(np.transpose([photo_ids, business_ids]), columns=['photo_id', 'business_id']),
                      train_y.loc[business_ids, :].reset_index(drop=True),
                      pd.DataFrame(fc7_outs, columns=FC7_COLS)], axis=1)
    rows.index = photo_ids
    processed += len(paths)
    print(processed)
    
    return rows


print("Reading photo_to_biz.")
photo_to_biz = pd.read_csv('../yelp-data/train_photo_to_biz_ids.csv', header=0, index_col='photo_id')
print("Reading train_y.")
train_y = prep_y(pd.read_csv('../yelp-data/train.csv', header=0, index_col='business_id'))
train_y.columns = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating',
                   'restaurant_is_expensive', 'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']


print("Reading photos_paths.")
photos_paths = sorted([photos_dir_path + filename for filename in os.listdir(photos_dir_path)])

# pool_size = 3

# print("Starting Pool of {pool_size} workers.".format(pool_size=pool_size))



# def pool_to_df(path):
#     return to_df(photo_to_biz, train_y, path)

#pool = Pool(processes=pool_size)

print("Starting conversion.")
import time
start = time.time()
#list_result = pool.imap_unordered(pool_to_df, photos_paths[:50])
list_result = [to_df(photo_to_biz, train_y, photos_paths[i:i+32]) for i in range(0, int(len(photos_paths) * d['fraction']), 32)]
end = time.time()
print(end - start)

print("Starting concatenation.")
result = pd.concat(list_result)

print("Starting saving to {output_path}".format(output_path=output_path))
result.to_hdf(output_path, 'fc7')



# for url in image_urls:
#     try:
#         rawim, im = prep_for_network(resize_and_crop(fetch_img(url)))
#
#         prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
#         top5 = np.argsort(prob[0])[-1:-6:-1]
#
#         plt.figure()
#         plt.imshow(rawim.astype('uint8'))
#         plt.axis('off')
#         for n, label in enumerate(top5):
#             plt.text(250, 70 + n * 20, '{}. {}'.format(n+1, CLASSES[label]), fontsize=14)
#     except IOError:
#         print('bad url: ' + url)

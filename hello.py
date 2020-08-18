# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# %tensorflow_version 2.x

import os
import sys
import math
import numpy as np
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
from tqdm.notebook import tqdm, trange
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
from matplotlib.patches import Rectangle, PathPatch
from tensorflow import keras as K
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import streamlit as st
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
import matplotlib.animation as animation
Writer = animation.FFMpegWriter(fps=30, codec='libx264') # Or 
from PIL import Image

from streamlit.elements import image_proto

image_proto.MAXIMUM_CONTENT_WIDTH = 750
# import python files

sys.path.append('/Users/nhamquochung/final-project/model-python-implementation')
import dataset, models, utils

st.markdown('# Children Drawing Website with SketchRNN!')

image = Image.open('presentation/opening.png')

st.image(image, caption="Based on Google's Quick, Draw! Dataset", use_column_width=True)

st.markdown('# What is SketchRNN?')

st.image('presentation/sketchrnn.png')

st.markdown('''
## Sequence-to-Sequence Variational Autoencoder
### Encoder: 
- Takes a sketch as input, output a latent vector z. 
- Sketch sequence S and S_reverse into a bi-directional RNN to obtain 2 hidden states.
- Concatenate 2 hidden states into h.
- Project h into two vectors mu and sigma.
- Construct a non-deterministic latent vector z that is conditional on the input sketch.
''')
st.image('presentation/z.png')
st.markdown('''
### Decoder: 
- Based on z, Construct initial states [h0, c0] with.''')
st.image('presentation/h0.png')
st.markdown('''
- Each step i, feed previous point S_i-1 and latent vector z. Output at each time step are parameters for a probability distribution of the next data point S_i.''')
st.image('presentation/yi.png')
st.markdown('''
- Vector y_i is broken down into parameters of the probability distribution of the next data point.
### When to stop drawing?
- All sequences are generated to the sequence max length. 
- After training, sample sketches from the model. Sample an outcome S_i' at each time step and feed it as input to the next time step until i = max_seq_len.
- Control level of randomness by a temperature parameter. When temp -> 0, model becomes deterministic and will consist of most likely point in the probability density function. 
''')

# load data

st.markdown('# Load rabbit dataset:')

data_class = 'rabbit' #@param ["cat","eye","rabbit"]

# allow_pickle set to True and load .npz file
data = np.load('data/{}.npz'.format(data_class),encoding='latin1',allow_pickle=True)

st.markdown('## Data format exploration:')
data['train'][0]

# use cleanup method from dataset.py
data_train = [dataset.cleanup(d) for d in data['train']]
data_valid = [dataset.cleanup(d) for d in data['valid']]
data_test = [dataset.cleanup(d) for d in data['test']]

max_seq_len = max(map(len, np.concatenate([data_train, data_valid, data_test])))
scale_factor = dataset.calc_scale_factor(data_train)

st.markdown("## Max sequence length: `131`")
st.markdown("## Scale factor: `34.726692`")
# data_train

fig, ax = plt.subplots(figsize=(5,5))
utils.plot_strokes(ax, data_train[0])
st.markdown('## Then plot the first sketch in train data:')
st.pyplot()

# Samples from test set

n = [4, 10]

fig, ax = plt.subplots(n[0],n[1],figsize=(12, 5))

perm = np.random.permutation(range(len(data_test)))[:n[0]*n[1]]

for i, idx in enumerate(perm):
    x, y = i // n[1], i % n[1]
    utils.plot_strokes(ax[x][y], data_train[idx])
    ax[x][y].set(xlabel=idx)
    
utils.plt_show()
st.markdown('# Randomly plot 40 sketches in the test set:')
# st.write(perm)
st.pyplot() 

# Prep test dataset
ds_test = np.array([dataset.pad(dataset.normalize(d, scale_factor), max_seq_len) for d in data_test])
n_points = np.argmax(ds_test[:,:,4]==1,-1)
n_strokes = ds_test[:,:,3].sum(1)

st.markdown('# Prepare test data. Add padding in front and after to max sequence length:')
st.header('Analyse first test sketch:')
st.write(ds_test[0])
st.write('Number of points: ', n_points[0])
st.write('Number of strokes: ', n_strokes[0])

pmin, pmax = np.percentile(n_points,[1,99])
smin, smax = np.percentile(n_strokes,[1,99])

fig, ax = plt.subplots(1, 2, figsize=(12,4))

st.header('Overview of the entire test set:')
ax[0].hist(n_points[(pmin < n_points) & (n_points < pmax)], 20, alpha=0.8)
ax[0].set(xlabel="points", 
          title="Distribution by number of points")

ax[1].hist(n_strokes[(smin < n_strokes) & (n_strokes < smax)], 10, alpha=0.8)
ax[1].set(xlabel="strokes", 
          title="Distribution by number of strokes")

utils.plt_show()

st.pyplot()


## Filter invalid samples
ds_test = ds_test[n_points > 0]
n_strokes = n_strokes[n_points > 0]
n_points = n_points[n_points > 0]

# # Loading the model

st.markdown('# Loading the model:')

hps = {
    "max_seq_len": max_seq_len,
    'batch_size': 100,
    "num_batches": math.ceil(len(data_train) / 100),
    "epochs": 1,
    "recurrent_dropout_prob": 0.1,
    "enc_rnn_size": 256,
    "dec_rnn_size": 512,
    "z_size": 128,
    "num_mixture": 20,
    "learning_rate": 0.001,
    "min_learning_rate": 0.00001,
    "decay_rate": 0.9999,
    "grad_clip": 1.0,
    'kl_tolerance': 0.2,
    'kl_decay_rate': 0.99995,
    "kl_weight": 0.5,
    'kl_weight_start': 0.01,
}

st.header('Hyperparameters:')

code = '''
hps = {
    "max_seq_len": max_seq_len,
    'batch_size': 100,
    "num_batches": math.ceil(len(data_train) / 100),
    "epochs": 1,
    "recurrent_dropout_prob": 0.1,
    "enc_rnn_size": 256,
    "dec_rnn_size": 512,
    "z_size": 128,
    "num_mixture": 20,
    "learning_rate": 0.001,
    "min_learning_rate": 0.00001,
    "decay_rate": 0.9999,
    "grad_clip": 1.0,
    'kl_tolerance': 0.2,
    'kl_decay_rate': 0.99995,
    "kl_weight": 0.5,
    'kl_weight_start': 0.01,
}
'''
st.code(code, language='python')

sketchrnn = models.SketchRNN(hps)
sketchrnn.models['full'].summary()

st.header('Model summary: ')

image = Image.open('presentation/model.png')

st.image(image, caption='Model summary', use_column_width=True)

initial_epoch, initial_loss = 100, 0.06
checkpoint = os.path.join('weights/', 'sketch_rnn_{}_weights.{:02d}_{:.2f}.hdf5').format(data_class, initial_epoch, initial_loss)
sketchrnn.load_weights(checkpoint)

# plot a random sample sketch

st.markdown('# Sample based on temperature=0.3, unconditional with randomized latent vector:')

n = [5, 10]
temp = 0.3

fig, ax = plt.subplots(n[0],n[1],figsize=(12, 5))

for i in trange(n[0] * n[1]):
    strokes = sketchrnn.sample(temperature=temp)
    utils.plot_strokes(ax[i // n[1]][i % n[1]], utils.to_normal_strokes(strokes))
    
fig.suptitle("Random Samples From Latent Space (t={})".format(temp))

utils.plt_show(rect=[0, 0, 1, 0.95])

st.pyplot()

fig = plt.figure(figsize=(12, 5))

n = [5,11]

with tqdm(total=n[0]*n[1]) as pbar:
    for i in range(n[0]):
        z = np.random.randn(1, hps['z_size']).astype(np.float32)

        for j in range(n[1]):
            ax = plt.subplot(n[0], n[1], i * n[1] + j + 1)
            if j == 0:
                strokes = sketchrnn.sample(z=z,greedy=True)
            else:
                strokes = sketchrnn.sample(z=z,temperature=j*0.1)
            utils.plot_strokes(ax, utils.to_normal_strokes(strokes))
            ax.set(xlabel='greedy' if j == 0 else 't={:.1f}'.format(j*0.1))

            pbar.update(1)
            
fig.suptitle("Varying temperature: ")

utils.plt_show(rect=[0, 0, 1, 0.95])
st.markdown('# Sample from varying temperature, unconditional generation with randomized latent vector:')
st.pyplot()

fig = plt.figure(figsize=(12, 5))

n = 5

indices = np.random.permutation(range(len(ds_test)))[:n]

with tqdm(total=n*11) as pbar:
    for i in range(n):
        idx = indices[i]

        d = np.expand_dims(ds_test[idx],0)
        z = sketchrnn.models['encoder'].predict(d[:,1:])[0]

        ax = plt.subplot(n, 11, i * 11 + 1)
        utils.plot_strokes(ax, utils.to_normal_strokes(d[0]))
        ax.set(xlabel=idx)

        pbar.update(1)

        for j in range(1,11):
            strokes = sketchrnn.sample(z=z, temperature=j*0.1)

            ax = plt.subplot(n, 11, i * 11 + j + 1)
            utils.plot_strokes(ax, utils.to_normal_strokes(strokes))
            ax.set(xlabel='t={:.1f}'.format(j*0.1))

            pbar.update(1)

utils.plt_show()
st.markdown('# Sample from varying temperature, conditional generation with latent vector from input sketch:')
st.pyplot()

n = 4

indices = np.random.permutation(range(len(ds_test)))[:n * 2]

fig, ax = plt.subplots(n, 13, figsize=(12, 5))

with tqdm(total=n*11) as pbar:
    for i in range(n):
        idx1, idx2 = indices[i*2:i*2+2]

        d1, d2 = ds_test[idx1], ds_test[idx2]
        z1 = sketchrnn.models['encoder'].predict(d1[np.newaxis,1:])[0].squeeze()
        z2 = sketchrnn.models['encoder'].predict(d2[np.newaxis,1:])[0].squeeze()

        utils.plot_strokes(ax[i][0], utils.to_normal_strokes(d1))
        ax[i][0].set(xlabel=idx1)

        for j in range(11):
            z = np.expand_dims(utils.slerp(z1,z2,j*0.1),0)
            strokes = sketchrnn.sample(z=z, greedy=True)
            utils.plot_strokes(ax[i][j+1], utils.to_normal_strokes(strokes))
            ax[i][j+1].set(xlabel='$t={:.1f}$'.format(j*0.1))
            
            pbar.update(1)

        utils.plot_strokes(ax[i][-1], utils.to_normal_strokes(d2))
        ax[i][-1].set(xlabel=idx2)
        
fig.suptitle("Interpolation Between Samples")
st.markdown('# Interpolate Between Samples:')

utils.plt_show(rect=[0, 0, 1, 0.95])

st.pyplot()

batch_size = 100
n = len(ds_test)

ppxs = np.zeros(n)

for i in trange(n // batch_size + 1):
    batch = ds_test[i*batch_size:(i+1)*batch_size]
    seq_lens = np.argmax(batch[:,:,4]==1,-1)
    
    enc_in, dec_in = batch[:,1:], batch[:,:-1]
    outputs = sketchrnn.models['full']([enc_in, dec_in])[0]
    loss = models.calculate_md_loss(enc_in, outputs)
    
    ppxs[i*batch_size:(i+1)*batch_size] = tf.math.reduce_sum(tf.squeeze(loss),-1) / seq_lens

norm = mpl.colors.Normalize(vmin=ppxs.min(), vmax=ppxs.max())
cmap = mpl.cm.get_cmap('RdYlBu')

# fig, ax = plt.subplots(1, 2, figsize=(12,5))

# ax[0].plot(n_strokes - (np.random.rand(len(n_strokes)) - .5), ppxs, 'bo',alpha=0.3)
# ax[0].set(xlabel="strokes", ylabel="perplexity",
#           title="perplexity vs strokes")

# ax[1].plot(n_points - (np.random.rand(len(n_points)) - .5), ppxs, 'bo',alpha=0.3)
# ax[1].set(xlabel="points", ylabel="perplexity",
#           title="perplexity vs points")

# utils.plt_show()

# st.pyplot()

n = [4, 10]

fig, ax = plt.subplots(n[0],n[1],figsize=(12, 5))

perm = np.argsort(ppxs)[:n[0]*n[1]]

for i, idx in enumerate(perm):
    x, y = i // n[1], i % n[1]
    utils.plot_strokes(ax[x][y], utils.to_normal_strokes(ds_test[idx]), ec=cmap(norm(ppxs[idx])))
    ax[x][y].set(xlabel='{}\n({:.4f})'.format(idx, ppxs[idx]))
    
fig.suptitle("Best Samples")
    
utils.plt_show(rect=[0, 0, 1, 0.95])
st.markdown("# Best samples with highest probabilities:")
st.pyplot()

n = [4, 10]

fig, ax = plt.subplots(n[0],n[1],figsize=(12, 5))

perm = np.argsort(ppxs)[-n[0]*n[1]:][::-1]

for i, idx in enumerate(perm):
    x, y = i // n[1], i % n[1]
    utils.plot_strokes(ax[x][y], utils.to_normal_strokes(ds_test[idx]), ec=cmap(norm(ppxs[idx])))
    ax[x][y].set(xlabel='{}\n({:.4f})'.format(idx,ppxs[idx]))
    
fig.suptitle("Worst Samples")
    
utils.plt_show(rect=[0, 0, 1, 0.95])
st.markdown("# Worst samples with lowest probabilities:")

st.pyplot()

n = 4
indices = shuffle(np.argsort(ppxs)[:50])[:n]

ds = ds_test[indices]
zs = sketchrnn.models['encoder'].predict(ds[:,1:])[0]

fig, ax = plt.subplots(n, 13, figsize=(12, 5))

with tqdm(total=n * 11) as pbar:
    it = range(n)
    for s, t in zip(it, np.roll(it,-1)):
        d1, d2 = ds[s], ds[t]
        z1, z2 = zs[s], zs[t]

        utils.plot_strokes(ax[s][0], utils.to_normal_strokes(d1))
        ax[s][0].set(xlabel=indices[s])
        
        for j in range(11):
            z = np.expand_dims(utils.slerp(z1,z2,j*0.1),0)
            strokes = sketchrnn.sample(z=z, greedy=True)
            utils.plot_strokes(ax[s][j+1], utils.to_normal_strokes(strokes))
            ax[s][j+1].set(xlabel='$t={:.1f}$'.format(j*0.1))
            pbar.update(1)

        utils.plot_strokes(ax[s][-1], utils.to_normal_strokes(d2))
        ax[s][-1].set(xlabel=indices[t])
        
utils.plt_show()

# st.pyplot()

# frames_per_inter = 60

# fig, ax = plt.subplots(figsize=(3,3))

# z = sketchrnn.sample(z=zs[np.newaxis,0], greedy=True)
# patch = utils.plot_strokes(ax, utils.to_normal_strokes(strokes))
# path = patch.get_path()

# def animate(strokes):
#     newPath = utils.make_mpl_path(utils.to_normal_strokes(strokes))
#     path.vertices, path.codes = newPath.vertices, newPath.codes
#     return [patch]

# def frames(n):
#     nz = len(zs)
#     for i, (z1, z2) in enumerate(zip(zs,np.roll(zs,-1,0))):
#         for t in np.linspace(0,1,n):
#             utils.update_progress((i+t)/nz)
#             z = np.expand_dims(utils.slerp(z1, z2, t), 0)
#             strokes = sketchrnn.sample(z=z, greedy=True)
#             yield strokes

# anim = FuncAnimation(fig, animate, frames=frames(frames_per_inter), 
#                      interval=30, save_count=len(zs)*frames_per_inter, blit=True)

# plt.close(anim._fig)
# anim.save('rabbits_loop.gif', writer='pillow', fps=30)
# # st.write(HTML(anim.to_html5_video()))

st.image("presentation/rabbits_loop.gif")

st.image('presentation/image4.png')
st.image('presentation/image1.png')
st.image('presentation/image2.png')
st.image('presentation/image3.png')


st.image('presentation/diagrams_loops.gif')
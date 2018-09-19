# Audio-Classification based on ANNs
## Procedure
A neural network is trained, based on images of the spectogram of audio-signals.
The trained network can then be tested on preferably other audio-signals, by loading it
into a UI-Helper program, that classifies the signal and visualizies the audio-stream and its
prediction.<br>
**Attention**: Take care that at every step in your procedure the dimensions of the spectogram, your network and later
at visualizing them, are the same.

## Spectograms
At first there needs to be the data. This can be accomplished by fragmenting a complete audio-stream
into small bits of spectograms.<br>
At first create the complete spectogram with the method *audio_to_complete_spectogram* in *create_spectogram.py* 
in the root folder. This will return a numpy-array of the complete spectogram.<br>
Then subdivide the spectogram in smaller parts as you need them and same them as images to your file-system.<br>
(The whole procedure can be seen in the main-method of *create_spectogram.py*)

## Training
You can train your neural network based your chosen architecture with the created spectograms (see existing architectures below).<br>
In most cases you will need to have (at least some) labelled data (You may have a look at *load_data(...)* from *distinguish.py*
at your root-folder).

## Visualizing
View your result by loading and classifying an audio-stream into the *PlayerUI* class from *player_ui.py* at your root
folder.<br>
The player loads a given audio-stream and classifies each time-step with the trained network. It then gives the possibility
to play the stream and at the same time it shows the classification with a red or green square (red: noise, green: speech)
and its certainty about it.<br>
<br>
![PlayerUI Screenshot](PlayerUI.png?raw=true "PlayerUI")
<br><br>
See the main-method in *player_ui.py* for more details.

## Existing Architectures
The existing model can be found in the */models* subfolder.<br>
A helpful overview of ANN-architectures can be found at: [ANN-Zoo](http://www.asimovinstitute.org/neural-network-zoo/)

### VanillaConv
This is the most simple architecture. It models a simple convolutional neural network and passes the input image
forward to 2 single neurons that match "noise" and "speaker". The network is then trained with labelled data to learn
the features of spectograms.<br>
The problem here is that a huge amount of labelled data is needed to create a general classifier for different audio
streams.

### AutoEncoder
The AutoEncoder first encodes the input image to a bottleneck-layer and then decodes this layer back the original image
via transposed convolution. The intention here is that the network learns to compress the input image into the most
important features of spectograms (that is: what is minimally needed to reconstruct a spectogram from a simple 
one-dimensional feature-vector).<br>
This bottleneck-layer is then used to train a readout-layer that maps the single one-dimensional layer to 2 single neurons
that stand for "noise" and "speaker". For this training a set of labelled data is needed. But compared to the 
Vanilla-Convolutional method this should be a lot fewer, because the readout-layer (hopefully) just has to learn "where" the
information needed for the classification stands in the bottleneck-layer.<br>
At inferencing, the network then needs a forward pass from the input-layer through the encoder, up to the readout-layer.
The complete decoder isn't needed here anymore.

### RNNs
This is not implemented yet, but the idea of using a recurrent neural network comes from limitations of convolutional
neural networks.<br>
CNNs have a limited spatial scope of their features, that has to be enlarged through deeper network structures. To avoid
this, you could use time-dependent architectures, that (in our case horizontally) takes input vectors, and "memorizes"
the last *n* time-steps in order classify the spectogram.<br>
This can even be trained unsupervised with LSTMs (see: [Unsupervised sentiment neuron](https://blog.openai.com/unsupervised-sentiment-neuron/)).<br>
After training you would then have a readout-layer again, that interprets the cell-state of the LSTM-Cell.
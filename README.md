# Annotation (or segmentation) of the electrocardiogram (ECG) with a long short-term memory neural network. 
Here, I experimented with annotating peaks of the ECG using a recurrent neural network in tensorflow's Keras.
In the beginning I struggled a bit to get the input/output right, which had to do with the way I tried to format ECG-peaks (as a sparse vector containing peaks (1) vs no peaks (0)). Aproaching it as a semantic segmentation problem (e.g. Seq2Seq) solved it for me. 
<br>It seems to work well on the QT database of physionet, but there are some cases that it has never seen where it fails; I haven't played with augmenting the ecgs yet.
 
[5oct 2019]
Since posting this 3 years ago, I noticed several publications using the (exact) same principle:
- Matlab: https://www.mathworks.com/help/signal/examples/waveform-segmentation-using-deep-learning.html
- IEEE: https://ieeexplore.ieee.org/document/8333406


## Model

```
model = Sequential()
model.add(Dense(32,W_regularizer=regularizers.l2(l=0.01), input_shape=(seqlength, features)))
model.add(Bidirectional(LSTM(32, return_sequences=True)))#, input_shape=(seqlength, features)) ) ### bidirectional ---><---
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu',W_regularizer=regularizers.l2(l=0.01)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(dimout, activation='softmax'))
adam = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']) 
```

## Getting Started
- Download ECG data using something like <code>wget -r -l1 --no-parent https://physionet.org/physiobank/database/qtdb/</code>
- Run, with as the first argument the directory where the ECG data is stored; or set <code>qtdbpath</code>.

## Output
A 2 lead ECG, the colors indicate P-Pend(yellow),Pend-Q(green), Q-R(red),R-S(purple),S-Tend(brown),Tend-P(pink). Training took about an hour on 6 cores. 
![example output](./images/example.png)
- Colors at the top show true annotations
- Colors at the bottom show predicted annotations

## Dependencies
I haven't got a list of all dependencies; but use this:  
- wfdb 1.3.4 ( not the newest >2.0, thanks BrettMontague); pip install wfdb==1.3.4

---
layout: post
title: "Why cannot predict Bitcoin price with vanilla Machine Learning"
description: "I started to work on time series on stock market for fun"
tags: bitcoin experiment machine learning market stock
minute: 7
---

> *This post was originally published on erogol.com (hosted on DigitalOcean) and recovered from the [Wayback Machine](https://web.archive.org) after the original server was lost. Some formatting or images may differ from the original.*

I started to work on time series on stock market for fun. I like to see if an AI bot trading without a manual help is possible or is just a luring dream. Lately, I read a lot about the topic raging from traditional financial technical analysis to ML to predict the market. What I see at the ML front is **m****any people claim to use ML with success. What I really experience is they have false conclusion without a real experimental basis with inaccurate interpretations. And the worse, many other people try to replicate their results (like me) and waste a lot of time.  I like to show here why I think vanilla ML is not possible at to predict market price, here Bitcoin in particular.**

> This post is an excerpt from a [small part](https://github.com/erogol/Prebitation/blob/master/11-%20Predict%20move.ipynb) of my [repo](https://github.com/erogol/Prebitation). Feel free to visit. There are many other experiments.

This work illustrates a simple supervised setting where a ML model predicts the next market move given the current state. It seems pretty easy but nah :).

We have two main assumptions  generally deemed to be true in market literature.

* All information describing the market is hidden under the price values.
* We go Semi-Markovian, meaning each prediction only depends on the present state.

Now, what we do here is very simple. Given the state as **High, Low, Open, Close** price values of the present step we like to predict the price move at the next step which is categorized as **Up, Down or Same.**

```python
DATA_PATH = "../data/bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2017-10-20.csv.csv"
df = read_data(DATA_PATH, 3)

df_feats = compute_features(df)
df_feats.dropna(inplace=True)
df_feats
```

Here we read Bitcoin price history downloaded from [here](https://www.kaggle.com/mczielinski/bitcoin-historical-data/data) into a Pandas dataframe, convert any row to a difference from the previous time step and drop None rows. That is, each row is a difference btw the time t and time (t-1) for each columns.

```python
# Validation split
train_split_point = time.mktime(datetime.datetime.strptime('2016-1-1 00:00:00', "%Y-%m-%d %H:%M:%S").timetuple())
split_point = time.mktime(datetime.datetime.strptime('2017-7-1 00:00:00', "%Y-%m-%d %H:%M:%S").timetuple())

df_train = df_feats[np.logical_and(df_feats['timestamp'] &amp;lt; split_point, df_feats['timestamp'] &amp;gt; train_split_point)]
df_test = df_feats[df_feats['timestamp'] &amp;gt; split_point]

print(df_train.shape)
print(df_test.shape)
```

Split the data into train and test by taking the date 2017-7-1 is the split point. So we use the market data after 2017-7-1 as the test set. That gives us 262541 steps for training and 53319 steps for testing.

```python
y_train, label_names = compute_labels(df_train)
y_test, _ = compute_labels(df_test)

check_labels(y_test.argmax(axis=1), df_test['close'].values)
check_labels(y_train.argmax(axis=1), df_train['close'].values)

assert y_train.shape[0] == df_train.shape[0]
assert y_test.shape[0] == df_test.shape[0]

X_train = df_train.iloc[:, -4:].values
X_test = df_test.iloc[:, -4:].values

assert y_train.shape[0] == X_train.shape[0]
assert y_test.shape[0] == X_test.shape[0]
```

Compute labels for each time step (t) as **Up, Down or Same**. If the label is Up, the price is predicted to increase at time (t+1).

Let's define the magic box with [Keras](https://keras.io/). This is a basic 4 layers fully connected network. You can play around the architecture as you like for your run.

```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(32, activation = 'tanh', input_dim = 4))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'tanh'))
model.add(Dropout(0.1))
model.add(Dense(32, activation = 'tanh'))
model.add(Dropout(0.1))
model.add(Dense(3, activation = 'softmax')) 
# out shaped on df_Yt.shape[1]
model.compile(loss='categorical_crossentropy', optimizer='adam', 
metrics=['accuracy'])
```

Train the model and enjoy the progress bar 🙂

```python
batch_size = 512 # Total 'blocks/snapshot' in a day
epochs = 1000

model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=batch_size, epochs=1000)
```

```python
Epoch 27/1000
262541/262541 [==============================] - 3s - loss: 0.8915 - acc: 0.5028 - val_loss: 0.9344 - val_acc: 0.4627
Epoch 28/1000
177152/262541 [====================..........] - ETA: 1s - loss: 0.8910 - acc: 0.5055
```

This is when I stop the learning. You should also see something similar.

```python
#if we always predict UP

precision recall f1-score support

timestamp 0.4694 1.0000 0.6389 25027
high 0.0000 0.0000 0.0000 22390
low 0.0000 0.0000 0.0000 5902

avg / total 0.2203 0.4694 0.2999 53319
```

Before we see the model performance, first we measure the baseline values. Considering the Bitcoin craze, If we **always predict UP**we already get **~0.22 accuracy**.

```python
#random prediction

precision recall f1-score support

timestamp 0.4732 0.3385 0.3947 25027
high 0.4262 0.3354 0.3754 22390
low 0.1112 0.3353 0.1670 5902

avg / total 0.4134 0.3368 0.3614 53319
```

**Random prediction** also obtains **~0.41 accuracy.** Now measure the model performance and see if we get something better.

```python
precision recall f1-score support

UP 0.4999 0.6983 0.5826 25027
DN 0.4540 0.2735 0.3413 22390
FLAT 0.3840 0.3168 0.3472 5902

avg / total 0.4678 0.4777 0.4552 53319
```

We obtain **0.47 accuracy** which is better than random and shows our model is keen to learn something. **Most of the people stop** here and believe that things gonna work but No!! it is not done yet.

Let's plot the predictions and see what  actually goes wrong. What we see here is the color coding of our prediction at each time step. **Green is Up, blue is Same and red is Down**.

![](https://web.archive.org/web/2020/http://https://web.archive.org/web/2020/http://erogol.com/wp-content/uploads/2017/12/btc_preds.png)

Very large plot and please open it on another window

The broken thing here, if we look carefully, the model only predicts what we have at the previous time step. If price stayed the same, it predicts Blue. If price was up before, it predicts Green and so on. With a model like this, it is normal to measure good accuracy since it is natural to expect Up move, if it went Up previously. It is a good catch for a Kaggler but not a trader.

That I can say, trained model is not generalizing the knowledge to help us but memorizing basic rules which makes it useless in a real-life case since what we like to have from a ML model is to signal sudden changes and see beyond the horizon cannier than us.

What I like to pin here is **not that ML is useless** on this problem. ML is definitely helpful with a more advance constructs. **Just don't expect to download data, train the model and be rich** :).

If you are really interested in using ML in trading, I suggest you to start from the basic and initially use ML for supporting signals. It might only use traditional indicators commonly used by experienced trades. However, do not rely on ML from the start and use it as a side-kick.

**Note that**, I try to keep things simple here but you might like to include many other features like financial indicators using the great library [TA-Lib](https://www.ta-lib.org/).  You can use any other vanilla supervised model. Or you can try to predict real price change by converting the problem into Regression. I assure you the result will be the same.

**What about RNN?** I should also point out that **RNN** (or LSTM, GRU) has far worse memorization problem. If you **train RNN for regressing** the relative price change, what it does is predicting a small variance over a previous time step price. Again, this gives satisfying model performance as Mean Squared Error is the basis but has no use.  Although this is a solution proposed by many blog posts, I once again assure you that RNN does not work too.

**Last remarks**, I believe ML has a huge playground especially at the newly emerged Crypto market for two main reasons. The first, since many people are just new in trading and they tightly follow well studied buy/sell patterns. So that can be learned by a ML model. The second, cryto market is a wild game. Things are so volatile. Things  go up 100%  or down 200% over a night. It is great opportunity for good traders but it is not possible to eye all the market with such volatility. So this is just a great reason to use AI to help us and enhance what we see on the horizon.

Pls let me know what you think.  Also feel free to ping me if you have something new or you like **AI based trading**. I personally start to use ML to do what I propose above. We can enjoy it together. Best!!

[![Share](https://static.addtoany.com/buttons/favicon.png)](https://www.addtoany.com/share)

### Related posts:

1. [Stochastic Gradient formula for different learning algorithms](http://www.erogol.com/stochastic-gradient-formula-for-different-learning-algorithms/ "Stochastic Gradient formula for different learning algorithms")
2. [NegOut: Substitute for MaxOut units](http://www.erogol.com/negout-substitute-for-maxout-units-2/ "NegOut: Substitute for MaxOut units")
3. [Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/ "Paper Review: Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?")
4. [Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?](http://www.erogol.com/paper-review-convergent-learning-different-neural-networks-learn-representations/ "Paper review: CONVERGENT LEARNING: DO DIFFERENT NEURAL NETWORKS LEARN THE SAME REPRESENTATIONS?")
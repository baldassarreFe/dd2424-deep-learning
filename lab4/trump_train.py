import functools
import json
import re
import time
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from datasets import InputTargetSequence
from initializers import Xavier, Zeros
from network import CharRNN
from rnn_optimizers import RnnAdaGrad
from utils import save_char_rnn

tweet_length = 140
tweet_full = 141
break_tweet_every = 14


def load_all_tweets():
    """
    :return: a list of all tweets, each one formatted as in `sanitize_and_pad`
    """
    json_path = 'sources/trump'
    json_files = [join(json_path, f) for f in listdir(json_path)
                  if isfile(join(json_path, f)) and f.endswith('.json')]
    all_tweets = []
    for j in json_files:
        with open(j) as f:
            loaded = json.load(f)
        all_tweets.extend([sanitize_and_pad(t['text']) for t in loaded])
    return all_tweets


def sanitize_and_pad(tweet_text):
    """
    Processes a tweet so that it contains only ascii characters, removes all
    internal newlines, changes the web-encoded '&' '>' '<' to their ascii char,
    pads to 140 characters adding spaces if needed and adds '\0' as starting
    character (a <start of tweet> marker)
    :param tweet_text:
    :return:
    """
    return '\0' + re.sub(r'[^\x00-\x7F]+', ' ', tweet_text) \
        .replace('&amp;', '&') \
        .replace('&lt;', '<') \
        .replace('&ge;', '>') \
        .replace('\n', '') \
        .replace('\r', '') \
        .ljust(140)


def encode_tweets(tweet_list):
    char_sequence = list(''.join(tweet_list))
    label_encoder = preprocessing.LabelBinarizer()
    label_encoder.fit(list(set(char_sequence)))
    np.savez_compressed('sources/trump/tweet_chars',
                        labels=label_encoder.classes_)
    encoded_tweets = label_encoder.transform(char_sequence)
    return encoded_tweets, label_encoder


def tweet_subsequences(encoded_tweets):
    for i in range(0, encoded_tweets.shape[0] - tweet_full, tweet_full):
        tweet = encoded_tweets[i:i + tweet_full]
        yield [InputTargetSequence(
            input=tweet[j:j + break_tweet_every],
            output=tweet[j + 1:j + break_tweet_every + 1]
        ) for j in range(0, tweet_length, break_tweet_every)]


def setup_plot():
    plt.plot([])
    plt.grid(True, which='major', color='k', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', color='k', linestyle='-', alpha=0.1)
    plt.minorticks_on()
    plt.title('Evolution of cost over the number of tweet pieces seen')
    plt.xlabel('Tweet pieces seen')
    plt.ylabel('Smoothed cost')
    plt.ion()


def report_callback(e, i, start, tweet_per_epoch, opt, label_encoder):
    first_char = np.squeeze(label_encoder.transform(['\0']))
    initial_state = np.zeros(opt.rnn.state_size)
    seq, _ = opt.rnn.generate(first_char, initial_state, tweet_length)
    gen = ''.join(label_encoder.inverse_transform(seq)).strip()
    print('Completed epochs {} tweets {}/{} '
          'cost {:.2f} elapsed {:.0f}s:\n{}\n'
          .format(e, i, tweet_per_epoch, opt.smooth_costs[-1],
                  time.time() - start, gen))
    plt.plot(opt.smooth_costs, 'b-')
    plt.pause(.05)


def epoch_callback(opt, e):
    save_char_rnn(opt.rnn, 'trump_{}'.format(e))
    plt.axvline(x=opt.steps, color='r')
    plt.pause(.05)


def main():
    tweet_list = load_all_tweets()
    encoded_tweets, label_encoder = encode_tweets(tweet_list)
    total_chars, num_classes = encoded_tweets.shape

    rnn = CharRNN(
        input_output_size=num_classes,
        state_size=100,
        initializer_W=Xavier(),
        initializer_U=Xavier(),
        initializer_V=Xavier(),
        initializer_b=Zeros(),
        initializer_c=Zeros()
    )
    opt = RnnAdaGrad(rnn, 0.1, stateful=True, clip=5)

    setup_plot()
    tweets_as_epochs = list(tweet_subsequences(encoded_tweets))
    total_tweets = len(tweets_as_epochs)
    report = functools.partial(report_callback, start=time.time(), opt=opt,
                               tweet_per_epoch=total_tweets,
                               label_encoder=label_encoder)
    for e in range(10):
        indexes = np.arange(total_tweets)
        np.random.shuffle(indexes)
        for i, idx in enumerate(indexes):
            if i > 0 and i % 2000 == 0:
                report(e, i)
            opt.train(tweets_as_epochs[idx])
        epoch_callback(opt, e)

    plt.savefig('plots/trump.png')


if __name__ == '__main__':
    main()

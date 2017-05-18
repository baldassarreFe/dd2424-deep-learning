# Tweets

## Preaparation

- remove \n\r chars
- replace &gt; chars with their ascii
- pad sequence with spaces
- start of tweet is marked by char \0 and null state

## Training

- split a 140+1 sequence in 10 chunks of 14 chars
- keep the state in between the sequence
- reset the state after every tweet
- for this reason, possibility to shuffle the tweets and avoid
  periodic fluctuations during training
- easily learned to generate @, # and url
- harder to make phrases with a sense because that's how tweets work

## Generation

- no need to warm up the rnn, it is enough to fee a \0 and a null state

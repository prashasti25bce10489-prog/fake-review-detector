# Fake Review Detector

Ordered food one too many times from places with 5 stars. Got cold, stale, just bad. Found out restaurants pay for fake reviews. So built this.

## What it does

Takes a review and tells you if it's real or fake. Nothing fancy.

## How

- Grabbed dataset from Kaggle. ~8000 reviews. Marked as real (CG) or fake (OR).
- Converted words to numbers using CountVectorizer.
- Threw it at Naive Bayes algorithm.
- 80% for training, 20% for testing.

## Does it work?

Got 82% accuracy. Not amazing but okay.

|                 | Model said Real | Model said Fake |
|-----------------|-----------------|-----------------|
| Actually Real   | 3412            | 604             |
| Actually Fake   | 829             | 3242            |

Translation:
- Caught 3412 real reviews correctly
- Caught 3242 fake reviews correctly
- Messed up 604 real ones (called them fake)
- Let 829 fake ones slide

## Words that give it away

Fake reviews use extreme words:

1. amazing
2. best
3. love
4. perfect
5. worst
6. terrible
7. must
8. highly

Real reviews are more balanced. Talk about food, service, delivery time, price.

## Run it

Python needed.

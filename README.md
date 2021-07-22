# Project Description

* This is the code for my project of evolving a Turtle to replicate an Image
* The Trainings take place in 3 different files:
    * `train_mona.py` contains the code for my first 2 experiment, the default
      config is feed-forward NEAT.
    * `train_random_turtle.py` contains the code for my 3rd and 4th experiment,
      the default config is feed-forward NEAT.
    * `train_turtle.py` contains the code for my final experiment, the default
      config is feed-forward NEAT.
* `checkpoint` contains the checkpoint for some successful experiment, I will only
  upload the best 10 genomes for each experiments.
* `config` contains the config of NEAT.
* Install Python 3.6 for stable installation

# Installation
```
conda create -n turtle python=3.6
pip install -r requirements.txt
```

# Training
```
python train_random_turtle.py
```

# Testing
* To test on a blank world
```
python test_turtle.py --img picture/heart.jpg

```
```
python test_turtle.py --img picture/mona.jpg

```
```
python test_turtle.py --img random

```
* To test on a random world
```
python test_turtle_erase.py --img picture/heart.jpg

```
```
python test_turtle_erase.py --img picture/mona.jpg

```
```
python test_turtle_erase.py --img random

```

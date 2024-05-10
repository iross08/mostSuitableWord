#!/bin/bash

# word recommendation app

# need flask integration

sudo docker build -t word_recommender .

sudo docker run -it --rm --name word_recommender word_recommender

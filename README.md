# NEUGO
![alt text](https://github.com/jinseokYeom/neugo/blob/master/neugo_banner.png "NEUGO")

## Overview
NEUGO is a simple neural network framework in Go. You can use this framework for 
fast prototyping that involves neural networks, simply by two steps: configure and
run. This framework does NOT include implementation of training methods; I had EAGO
in mind when I started this project. EAGO will train the neural nets through its
NE (NeuroEvolution) package.

## Algorithms
Following are available algorithms currently. More will be added with further
updates.
* Feedforward neural network

## Planned Algorithms
These are algorithms that are not implemented yet, but are planned to be added
in the future.
* Recurrent Neural Network (LSTM)
* Convolutional Neural Network
* Neural Turing Machine

## Installation
`go get github.com/jinseokYeom/neugo`

## Check out EAGO (Evolutionary Algorithms in Go)!
You can use my other package, EAGO, to train your neural network, very easily.
All it takes is configure and run!

## Note
The purpose of this package is for me to learn more about various kinds of neural
networks; I would appreciate any constructive feedbacks!

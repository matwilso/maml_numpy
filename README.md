# MAML in raw numpy

This is an implementation of vanilla Model-Agnostic Meta-Learning ([MAML](https://github.com/cbfinn/maml))
in numpy.  I made this to better understand the algorithm and what exactly it is doing.  I derived
the forward and backward passes following conventions from in [CS231n](http://cs231n.github.io/).
This turned out to be pretty interesting and I think it does help with understanding the algorithm,
because the logic is less hidden than in TensorFlow.  Like to define the meta-forward
pass, you need to embed an inner forward and backward pass and update step.  And then
for the meta-backward, you need to backprop all the way through it all.

I think this is actually a good way to understand MAML and what it is doing.

This is just a rough sketch to understand the algorithm better, so the code is not
well parameterized.  The forward and backward passes and number of layers are hard-coded.


**Table of contents**
- [Results](#results)
- [What is MAML?](#whatismaml)
- [Derivation](#derivation)


## Results
<a id="results"/>

I train on the sinusoid task from [Section 5.1](https://arxiv.org/pdf/1703.03400.pdf)
of the MAML paper.

These figures show MAML fine-tuning on 10 minibatches of 5, and against a baseline
that used joint training on the dataset (first plot), and against a random 
intialized network (second plot).

![numpy MAML sinusoid baseline](/assets/numpy/maml_baseline.png)
![numpy MAML sinusoid random](/assets/numpy/maml_random.png)


Here are the commands to the run the code:

Run 10k iterations and then save the weights to a file: <br>
```
python3 maml.py # train the MAML and baseline (joint trained) weights
```

Fine tune the network and plot results on sine task: <br>
```
python maml.py --test 1  
```


You can also do gradient check with:

```
python maml.py --gradcheck 1  
```


You can also try maml_2layer.py.  This has a single hidden layer and I found that 
it did not have enough representational capacity to solve this meta-learning
problem.  It is nearly the same as the maml.py file, except shorter and easier
to understand and doesn't work well.



## What is MAML?
<a id="whatismaml"/>
// TODO: maybe add explanation here about what MAML is



## Derivation
<a id="derivation"/>

Here is a diagram showing the derivation.
// TODO: make in inkscape






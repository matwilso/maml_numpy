# MAML in raw numpy

This is an implementation of vanilla Model-Agnostic Meta-Learning ([MAML](https://github.com/cbfinn/maml))
in raw numpy.  I made this to better understand the algorithm and what exactly it is doing.  I derived
the forward and backward passes following conventions from [CS231n](http://cs231n.github.io/).
This code is just a rough sketch to understand the algorithm better, so it works, but 
is not optimized or well parameterized.  
This turned out to be pretty interesting and I found it helps to see the algorithm 
logic without the backprop abstracted away by an autograd package like TensorFlow.

**Table of contents**
- [Results](#results)
- [What is MAML?](#whatismaml)
- [Derivation](#derivation)


<a id="results"/>

## Results

To verify my implementation, I test on the sinusoid task from [Section 5.1](https://arxiv.org/pdf/1703.03400.pdf)
of the MAML paper.  I train for 10k iterations on a training dataset of sine
function with randomly sampled amplitude and phase, 
and then fine-tune on 10 examples for a fixed amplitude
and phase.  After fine-tuning, I predict the value of the fixed sine function 
for 50 evenly distributed x values between (-5, 5), and plot the results
compared to the ground truth for pre-trained MAML, pre-trained baseline
(joint training), and a randomly initialized network.  

MAML                       |  Baseline (joint training)|  Random init
:-------------------------:|:-------------------------:|:----------:|  
![](/assets/maml.png)  |  ![](/assets/baseline.png) | ![](/assets/random.png)

Here are the commands to the run the code:

- Train for 10k iterations and then save the weights to a file: <br>
	```
	python3 maml.py # train the MAML and baseline (joint trained) weights
	```
- After training, fine-tune the network and plot results on sine task: <br>
	```
	python3 maml.py --test 1  
	```
- Run gradient check on implementation:
	```
	python3 maml.py --gradcheck 1  
	```


### Notes
These results come from using a neural network with 2 hidden layers.  I 
originally tried using 1 hidden layer because it was easier to derive, but I 
found that it did not have enough it did not have enough representational 
capacity to solve the sinusoid problem (see [Meta-Learning And Universality](https://arxiv.org/pdf/1710.11622.pdf) for more details on represenntational capacity of MAML).

So the `maml_1hidden.py` file is shorter and easier to understand, but does not produce good results.

<a id="whatismaml"/>

## What is MAML?

### Introduction

Model-Agnostic Meta-Learning (MAML) is a gradient based meta-learning algorithm.  For an
overview of meta-learning, see a blog post from the author [here](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/). 
Basically meta-learning tries to solve the problem of being able to learn 
quickly on new tasks by better incorporating past information from previous tasks.
It has some similar motivations to transfer learning, but better incentivizes for
quick adaptation.  One example of where meta-learning methods greatly 
outperform traditional methods is one-shot learning (e.g., given a single 
instance of a new object class like Segway, quickly adapt your model so that
you can effectively distinguish new images of Segways from other objects).

Unlike several other meta-learning methods (TODO: cite), MAML only uses 
feed-forward networks and gradient descent.  The interesting thing is how it 
sets up the gradient descent scheme to optimize the network for efficient 
fine-tuning on the meta-test set.
In standard neural network training, we use gradient-descent and backprop for 
training.  MAML assumes that you will use this same approach to quickly 
fine-tune on your task and it builds this into the training optimization.

MAML breaks the training into two phases: a **meta-traning phase** and a **fine-tuning phase**.  The meta-training phase is going to work to optimize the network parameters so that the fine-tune phase works extremely well — so that the network parameters will be sensitive to gradients and can
quickly adapt to solve tasks in the distribution.  The fine-tuning phase will just
run standard gradient descent using the weights that were produced in the meta-training phase (this looks pretty similar to fine-tuning on pre-trained ImageNet weights,
but it produces better results on meta-learning problems like one-shot learning).

### Meta-training
During meta-training, MAML splits the data into A and B examples.  The A example
will be used for an inner optimization (standard gradient descent), and the B 
examples will be used for an outer optimization.

The standard equation for gradient descent looks like this equation, where the
the gradient is taken with respect to the loss and this is used to update the
neural network weights (in this case to theta prime). 

![eq1](./assets/eq1.png)

We will use this at fine-tune time and so we want it to optimize it to do really
well. The approach that MAML chooses to do this is by optimizing the parameters
(theta) so that they will be in a position that they can be quickly fine-tuned
for a number of tasks in the task distribution.  Just a skip away from a good
solution to many of the tasks.  Why not just optimize to be good at those
tasks in the first place?  Well sometimes they can be mutually exclusive.  The
simplest example is the sinusoid, where this standard joint training approach
always predicts 0.


we want it to be really efficient. We
want it to be the case that after we run this fine-tuning step, the weights
will be in a good place to produce good results. We can't just keep them in
a good place for all tasks.  For example, the joint training solution for
the sinusoid task is always predicting 0. But we want to be able to quickly
update them so that they will be in a good location for our specific sinusoid
problem.  So how to optimize the weights so that this will happen.

We care about the performance after a gradient update, so we can just wrap
that in an outer optimization.  The inner optimization is the gradient
update, and the outer optimization is also gradient descent (or usually Adam).
You could have the outer optimization be evolution (they do this in this paper,
but they use way more compute and get about similar results, maybe worse https://arxiv.org/abs/1806.07917). Using gradient descent on the outer optimization looks like:

![eq2](./assets/eq2.png)

With the f theta prime, that is the updated weights.  We run gradient descent
such that after the update, the network is good at predicting new examples.
The inner update is done with A examples and the outer update is done with
B examples for the same task.  We basically use test error to train the network.
As you might be able to infer, this is a second order method that requires running backpropagation twice.


**Here is some pseudocode that matches Chelsea's implementation of MAML in TensorFlow:**

```python
weights = make_NN_weights() # make weights and biases

inputA, labelA, inputB, labelB = data.meta_split()

# forward pass of network using weights and A examples
netoutA = forward(inputA, weights)
lossA = loss_func(netoutA, labelA)

gradients = get_gradients(lossA) # w.r.t. weights

fast_weights = weights + -learning_rate * gradients   # gradient descent step on weights

netoutB = forward(inputB, fast_weights)
lossB = loss_func(netoutB, labelB)

# then you would plug this lossB in an optimize like Adam to optimize
# w.r.t. to the original weights.  fast_weights are basically just a temporary
# thing to see how gradient descent on the inner lossA led to update them.
# The only state that is retained between iterations of MAML are the 
# slow weights.
```

This seems like of a stange thing to do. But basically the effect is 


You can also run more than one step.  Maybe you would run the fast weights update
multiple times.

### MAML algorithm

In the simplest case, you can think of it as doing 2 forward passes. It does:
1. Forward pass with W
1. Backward pass to compute gradients dWa
1. Apply gradients dW (using SGD: W' <-- W - alpha\*dWa)
1. Another forward pass with W'
1. Backward pass through the whole thing to compute gradients dWb (NOTE: with respect to input weights W, not W'.  This is a second order derivative)
1. Apply gradients dW' (using Adam: W <-- W - alpha\*dWb)


### Fine-tuning

At the fine-tune stage, you have a set of meta-trained weights.  Now given a new task ---
say you want to train to predict new instances of samples drawn from a fixed sinusoid, given
only a few examples --- you feed examples and you fine-tune, using only the inner
gradient. You treat them like new A examples, and you keep track of the fast_weights
and you can now use those fast weights to predict new examples.

**Here is some pseudocode to illustrate how fine-tuning works and relates to training**
```
inputA, labelA = test_data

netoutA = forward(inputA, weights)
lossA = loss_func(netoutA, labelA)

gradients = get_gradients(lossA) # w.r.t. weights

fast_weights = weights + -learning_rate * gradients   # gradient descent step on weights


newInputAToPredictLabelFor = new example

prediction = forward(newInputAToPredictLabelFor, fast_weights)
```


### Notes
Beyond the scope of this README, Chelsea Finn and colleagues have done some 
interesting further work on extending MAML and applying it to robotics problems.
Particularly, being able to use a learned loss is extremely interesting.

Check out MIL and DAML.

<a id="derivation"/>

## Derivation

The below diagram shows the meta forward pass for MAML with a single inner
updated step.  By computing the gradients through this computational graph,
I determined the computations required for the meta backwared pass. I show
the computation for a one hidden-layer neural network for simplicity, but
in the code I use a two hidden-layer neural network.

NOTE: (dW2, db2, dW1, db1) are computed in the upper figure and passed to the lower
figure. Gradients are taken w.r.t. weights that are inputs to upper figure. I
use the approach from [CS231n](http://cs231n.github.io/).

**First (inner) forward and backward:**

![derivation](/assets/derivation.png)

**Inner gradient (SGD) update and second forward pass:**
![derivation2](/assets/derivation2.png)


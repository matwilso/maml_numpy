# MAML in raw numpy

This is an implementation of vanilla Model-Agnostic Meta-Learning ([MAML](https://github.com/cbfinn/maml))
in raw numpy.  I made this to better understand the algorithm and what it is doing.  I derived
the forward and backward passes following conventions from [CS231n](http://cs231n.github.io/).
This code is just a rough sketch to understand the algorithm better, so it works, but 
is not optimized or well parameterized.  
This turned out to be pretty interesting to see the algorithm 
logic without the backprop abstracted away by an autograd package like TensorFlow.

**Table of contents**
- [Results](#results)
- [What is MAML?](#whatismaml)
- [Derivation](#derivation)


<a id="results"/>

## Results

To verify my implementation, I test on the 1D sinusoid regression problem 
from [Section 5.1](https://arxiv.org/pdf/1703.03400.pdf) of the MAML paper (see
also the description of the problem in [Section 4](https://arxiv.org/pdf/1803.02999.pdf) of the Reptile).

I train for 10k iterations on a [dataset](utils/data_generator.py) of sine 
function input/outputs with randomly sampled amplitude and phase, and then 
fine-tune on 10 samples from a fixed amplitude and phase.
After fine-tuning, I predict the value of the fixed sine function 
for 50 evenly distributed x values between (-5, 5), and plot the results
compared to the ground truth for pre-trained MAML, pre-trained baseline
(joint training), and a randomly initialized network.  I find that
MAML is able to fit the sinusoid much more effectively.

MAML                       |  Baseline (joint training)|  Random init
:-------------------------:|:-------------------------:|:----------:|  
![](/assets/maml.png)  |  ![](/assets/baseline.png) | ![](/assets/random.png)

Here are the commands to the run the code:

- Train for 10k iterations and then save the weights to a file: <br>
	```
	python3 maml.py # train both MAML and baseline (joint trained) weights
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
originally tried using 1 hidden layer (see [`maml_1hidden.py`](maml_1hidden.py), 
because it was easier to derive, but I 
found that it did not have enough it did not have enough representational 
capacity to solve the sinusoid problem (see [Meta-Learning And Universality](https://arxiv.org/pdf/1710.11622.pdf) for more details on representational capacity of MAML).

<a id="whatismaml"/>

## What is MAML?

### Introduction

Model-Agnostic Meta-Learning (MAML) is a gradient based meta-learning algorithm.  For an
overview of meta-learning, see a blog post from the author [here](https://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/), and a good talk [here](https://youtu.be/i05Fk4ebMY0). 
Roughly meta-learning tries to solve sample-ineffiency problems in
machine learning.  It tries to allow models to learn 
quickly on new tasks by better incorporating past information from previous tasks.

Unlike [several](https://arxiv.org/abs/1611.02779) [other](https://openreview.net/forum?id=rJY0-Kcll) [meta-learning](https://arxiv.org/abs/1606.04474) [methods](https://arxiv.org/abs/1707.03141) which use RNNs, MAML only uses 
feed-forward networks and gradient descent.  The interesting piece is how it 
sets up the gradient descent scheme to optimize the network for efficient 
fine-tuning on the meta-test set.
In standard neural network training, we use gradient-descent and backprop for 
training.  MAML assumes that you will use this same approach to quickly 
fine-tune on your task and it builds this into the meta-training optimization.

MAML breaks the meta-learning problem into two phases: a **meta-traning phase** and a **fine-tuning phase**.  The meta-training phase optimizes the network parameters so that the fine-tune phase is more effective — so that the network parameters will be sensitive to gradients and can
quickly adapt to solve newly sampled tasks in the distribution.  The fine-tuning phase will just
run standard gradient descent using the weights that were produced in the meta-training phase, just like you would fine-tune a network for a task using e.g., pre-trained
ImageNet weights.  This process looks somewhat similar to transfer learning, but
is more general and produces better results on meta-learning problems like one-shot learning (where you are given a single instance of a new 
object class like electric scooter, and your model must quickly adapt so that
it can effectively distinguish new images of electric scooters from other objects).


### Meta-training
During meta-training, MAML draws several samples from a **task**, and splits them
into **A** and **B** examples. For example you could draw 10 (x,y) pairs from a sinusoid
problem and split them into 5 A and 5 B examples.  In this case each task is
defined by a fixed amplitude and phase of the sinusoid, but tasks can represent
more interesting variations, like what objects the robot should interact
with in [imitating a human demonstration](https://sites.google.com/view/daml).

Once we have sampled the A and B examples from the task, we will use the A
examples for an **inner optimization** (standard gradient descent),
and the B examples for **outer optimization** (gradient descent back through
the inner optimization).  At a high level: we will inner optimize on the A
examples, test the generalization performance on the B examples, and
meta-optimize on that loss (using gradient descent through the whole 
computation) in order to place the parameters in a good initialization
for quickly fine-tuning to many varied tasks.

For concretely how that is done, here is the algorithm logic and 
pseudocode that closely match the [TensorFlow
implementation](https://github.com/cbfinn/maml):

### MAML algorithm


**Algorithm logic (do this for many loops)**

1. Sample task T from distribution of possible tasks
1. Sample examples from T and split into A and B examples 
1. Network forward pass with weights W, using A examples
1. Backward pass to compute gradients dWa
1. Apply gradients dWa using SGD: W' <-- W - alpha\*dWa
1. Forward pass with temp weights W', using B examples this time
1. Backward pass through the whole thing to compute gradients dWb (NOTE: this gradient is with respect to input weights W, not W'.  This is a second order derivative and backprops through the B forward, the gradient update step, the A backward, and the A forward computations. Everything in the below [derivation diagrams](#derivation) is just the meta-forward pass.  This is backpropping through the whole thing, starting at pred_b)
1. Apply gradients dW' (using Adam: W <-- W - alpha\*dWb)


NOTE: You could also do batches of tasks at a time and sum the lossBs.

**Pseudocode that roughly matches Finn's implementation of [MAML in TensorFlow](https://github.com/cbfinn/maml):**

```python
weights = init_NN_weights() # neural network weights and biases

task_data = sample_task()

inputA, labelA, inputB, labelB = task_data.meta_split()

# forward pass of network using weights and A examples
netoutA = forward(inputA, weights)
lossA = loss_func(netoutA, labelA)

gradients = get_gradients(lossA) # w.r.t. weights

fast_weights = weights + -learning_rate * gradients   # gradient descent step on weights

netoutB = forward(inputB, fast_weights)
lossB = loss_func(netoutB, labelB)

# then you would plug this lossB in an optimizer like Adam to optimize
# w.r.t. to the original weights.  fast_weights are basically just a temporary
# thing to see how gradient descent on the inner lossA led to update them.
# The only state that is retained between iterations of MAML are the weights (not fast).
```

### Fine-tuning

At the fine-tune stage, you now have a set of meta-trained weights.  Given a 
new task, you can just run the inner optimization, keep track of the 
fast_weights, and then use them to predict new examples.  

**Pseudocode to illustrate how fine-tuning works and relates to training**
```
inputA, labelA = test_data()

netoutA = forward(inputA, weights)
lossA = loss_func(netoutA, labelA)

gradients = get_gradients(lossA) # w.r.t. weights

fast_weights = weights + -learning_rate * gradients   # gradient descent step on weights


prediction = forward(new_input_to_predict_label_for, fast_weights)
```


<a id="derivation"/>

## Derivation


The below diagram shows the meta-forward pass for MAML with a single inner
update step.  By computing the gradients through this computational graph,
I derived the computations required for the meta-backwared pass. I show
the computation for a single hidden-layer neural network for simplicity, but
in the code I use a two hidden-layer neural network.

NOTE: (dW2, db2, dW1, db1) are computed in the upper figure nd passed to the lower
figure. Gradients are backpropagated from the output all the way back through
both through to the upper figure. I use the approach from [CS231n](http://cs231n.github.io/).

**Inner forward and backward:**
![derivation](/assets/derivation.png)

**Inner gradient (SGD) update and second (outer) forward pass:**
![derivation2](/assets/derivation2.png)


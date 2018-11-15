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
defined by a fixed amplitude and phase of the sinusoid, but these can represent
more interesting tasks, like what the robot should do in a [robotic imitation problem](https://sites.google.com/view/daml).

Once we have sampled the A and B examples from the task, we will use the A
examples for an **inner optimization** (standard gradient descent),
and the B examples for **outer optimization**.

#### Inner optimization

Recall that the standard equation for gradient descent looks like this below 
equation, where the gradient is taken with respect to the loss and this is used to update the
neural network weights (in this case to theta prime). 

![eq1](./assets/eq1.png)

For MAML, we assume that we are going to use this simple gradient update
at fine-tune time, so we will want to somehow define our optimization
so that it will be be especially effective.  There are many ways we could do this,
and many ways that people have tried.
MAML's approach is to **meta-learn a good initialization** of 
theta, so that the parameters are in a good place in parameter space such
that they can quickly adapt with gradient descent to yield good performance on a task at fine-tune
time.  So how does it do this?
By baking the inner optimization into the outer optimization.

#### Outer optimization

This is where the A and B examples come in.
For the inner optimization, we run the A samples through the network and 
compute the gradient to optimize the network to be good at the task.  


We run the inner optimization using the A samples by passing the
A inputs through the network and optimizing A outputs and labels to train the network.
Then we are 



We care about the performance of the model after we do a gradient update, 
so we are going to run the inner optimization and then test it on new examples to see how
well it did and we will use these new test example labels to generate
a gradient signal to learn with.  And we are going to pass this
gradient signal back through the whole enchilada.  This is where
the A and B examples come in.  

Concretely, we 

we are going to
do the inner optimization step using A examples and then test the 
performance on the B examples and optimize all the way through the 
gradient descent step.  This will make MAML a second-order method.  We
take a gradient through the B forward and through the gradient update
computed with the A examples.

More concretely< 

so we can just wrap
that in an outer optimization.  

Basically do the inner gradient update using the A examples.  This is 
the same operation you will do at fine-tune time.  You are going to 
optimize the weights (theta) to do well on this and you are going to 
save them into a temporary weights.  In the MAML implementation, they
call these fast weights.  After you have the fast weights, you are
going to use them in another forward pass, this time using the B
examples. These B examples represent a test loss.  

We would like (and we expect we can reasonably assume a network to be able)
to do really well on these B examples because we have already seen the
A examples from the same task.  Performance on this B test pass is going
to measure how good our network is add learning quickly from these A
examples.  And we can define a loss in the same way



You would have liked to
do really well on these B examples because that would mean that your
step on the A examples was really good and helpful for learning 
about the task that they are both drawn from.

You really wish that you did well on those.  So what can we optimize 
so that this is more likely?  We can't change much about the gradient
descent computation, but we can change where the weights started before
we update them with our A data.  If they start in a good place, it would
only take a small nudge from the A examples to get them in the right place
for doing well on the test B examples.

So the trick we are going to do is optimize through the inner optimization.
We are going to optimize the slow weights that were input to the network.

This may be a bit confusing, and I think the easiest way to understand
the algorithm is to look at the pseudocode, which is more precise and 
quite simple. 




The inner optimization is the gradient
update, and for MAML, the outer optimization is also gradient descent (or more 
precisely, AdamOptimizer).
You could also have the outer optimization be evolution (they do this 
in [Meta-Learning by the Baldwin Effect](https://arxiv.org/abs/1806.07917), 
but they use way more compute and get about similar results).
Anyway, using gradient descent on the outer optimization looks like:






Why not just optimize to be good at those tasks in the first place?  Well 
sometimes they can be mutually exclusive.  The simplest example is the 
sinusoid, where this standard joint training approach always predicts 0, 
because this minimizes the expected loss when you have randomly sampled phases.


place in parameter space so that they can quickly be fine-tuned for a number
of tasks in the task distribution — just a skip away from a good solution to many of the tasks.  




TODO: this equation is too jarring.  Need to make equation intro nicer

![eq2](./assets/eq2.png)

With the f theta prime, that is the updated weights.  We run gradient descent
such that after the update, the network is good at predicting new examples.
The inner update is done with A examples and the outer update is done with
B examples for the same task.  We basically use test error to train the network.
As you might be able to infer, this is a second order method that requires running backpropagation twice.


### MAML algorithm


**Algorithm logic (do this for many loops)**

1. Sample task T from distribution of possible tasks
1. Sample examples from T and split into A and B examples 
1. Forward pass with weights W, using A examples
1. Backward pass to compute gradients dWa
1. Apply gradients dWa using SGD: W' <-- W - alpha\*dWa
1. Forward pass with temp weights W', using B examples this time
1. Backward pass through the whole thing to compute gradients dWb (NOTE: this gradient is with respect to input weights W, not W'.  This is a second order derivative and backprops through the B forward, the gradient update step, the A backward, and the A forward computations. Everything in the below [derivation diagrams](#derivation) is just the meta-forward pass.  This is backpropping through the whole thing, starting at pred_b)
1. Apply gradients dW' (using Adam: W <-- W - alpha\*dWb)

**Pseudocode that roughly matches Chelsea's implementation of [MAML in TensorFlow](https://github.com/cbfinn/maml):**

```python
weights = make_NN_weights() # neural network weights and biases

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

At the fine-tune stage, you have a set of meta-trained weights.  Now given a new task ---
say you want to train to predict new instances of samples drawn from a fixed sinusoid, given
only a few examples --- you feed examples and you fine-tune, using only the inner
gradient. You treat them like new A examples, and you keep track of the fast_weights
and you can now use those fast weights to predict new examples.

**Pseudocode to illustrate how fine-tuning works and relates to training**
```
inputA, labelA = test_data

netoutA = forward(inputA, weights)
lossA = loss_func(netoutA, labelA)

gradients = get_gradients(lossA) # w.r.t. weights

fast_weights = weights + -learning_rate * gradients   # gradient descent step on weights


newInputAToPredictLabelFor = new example

prediction = forward(newInputAToPredictLabelFor, fast_weights)
```


### Summary

This inference-time optimization seems especially interesting and probably a more
general thing that just the results/method of MAML.  It seems that it will
be powerful to allow models to optimize at execution time given the task.

(NOTE: also, Chelsea Finn and colleagues have done some 
interesting further work on extending MAML and applying it to robotics problems.
See [MIL](https://arxiv.org/abs/1709.04905) and [DAML](https://arxiv.org/abs/1802.01557))



<a id="derivation"/>

## Derivation


The below diagram shows the meta-forward pass for MAML with a single inner
update step.  By computing the gradients through this computational graph,
I derived the computations required for the meta-backwared pass. I show
the computation for a single hidden-layer neural network for simplicity, but
in the code I use a two hidden-layer neural network.

NOTE: (dW2, db2, dW1, db1) are computed in the upper figure and passed to the lower
figure. Gradients are backpropagated from the output all the way back through
both through to the upper figure. I use the approach from [CS231n](http://cs231n.github.io/).

**First (inner) forward and backward:**

![derivation](/assets/derivation.png)

**Inner gradient (SGD) update and second forward pass:**
![derivation2](/assets/derivation2.png)


# MAML in raw numpy

To understand Model-Agnostic Meta-Learning ([MAML](https://github.com/cbfinn/maml)) better, 
I derived the forward and backward passes for it and coded it up in numpy, following
conventions from [CS231n](http://cs231n.github.io/)

// TODO: maybe add explanation here about what MAML is

These are the results you can see after training it.  These show MAML
fine-tuning on 10 minibatches of 5, against a baseline that was joint
trained on the dataset (first plot), and against a random intialize
network (second plot).

![numpy MAML sinusoid baseline](/assets/numpy/maml_baseline.png)
![numpy MAML sinusoid random](/assets/numpy/maml_random.png)


Here is a diagram showing the derivation.
// TODO: make in inkscape



Here are the commands to the run the code:

```
python3 maml.py # train the MAML and baseline (joint trained) weights
```

This will run 10k iterations and then save the weights to a file.


```
python maml.py --test 1  # fine-tune the networks and plot the results on the sine function
```


You can also do gradient check with:

```
python maml.py --gradcheck 1  
```


----


You can also try maml_2layer.py.  This has a single hidden layer and I found that 
it did not have enough representational capacity to solve this meta-learning
problem.  It is nearly the same as the maml.py file, except shorter and easier
to understand and doesn't work well.





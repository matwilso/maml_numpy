# MAML in raw numpy

I hard-coded the full meta forward and backward passes of training, using a 3-layer neural
network.  This can solve the toy sinusoid meta-learning problem presented in the 
MAML paper.



```
python3 maml.py # train the MAML and baseline (joint trained) weights
```

This will run 10k iterations and then save the weights to a file.


```
python maml.py --test 1  # fine-tune the networks and plot the results on the sin function
```


You can also do gradient check with:

```
python maml.py --gradcheck 1  
```


----

These are the results you can see after training it.  These show MAML
fine-tuning on 10 minibatches of 5, against a baseline that was joint
trained on the dataset (first plot), and against a random intialize
network (second plot).

![numpy MAML sinusoid baseline](/assets/numpy/maml_baseline.png)
![numpy MAML sinusoid random](/assets/numpy/maml_random.png)



You can also try maml_2layer.py.  This has a single hidden layer and I found that 
it did not have enough representational capacity to solve this meta-learning
problem.  It is nearly the same as the maml.py file, except shorter and easier
to understand and doesn't work well.


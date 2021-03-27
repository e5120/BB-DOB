# BB-DOB
The Black-Box Discrete Optimization Benchmarks

## Setup
BB-DOB requires:
- Python >= 3.6

Install `BB-DOB` from the sources:

```
git clone https://github.com/e5120/BB-DOB
cd BB-DOB
pip install -r requirements.txt
pip install -e .
```

## Features
The search space is *D*-dimensional bit-strings **c** &in; {0,1}<sup>D</sup>.
- One-Max: <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(\boldsymbol{c})=\sum_{i=1}^Dc_i" />
- Two-Min: <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(\boldsymbol{c},&space;\boldsymbol{y})=\min(\sum_{i=1}^D|c_i-y_i|,\sum_{i=1}^D|(1-c_i)-y_i|)" />
  
  - **y** is *D*-dimensional bit-strings which is generated randomly in advance.

- Four-peaks: <img src="https://latex.codecogs.com/gif.latex?\inline&space;f(\boldsymbol{c})=\max(o(\boldsymbol{c}),z(\boldsymbol{c}))&plus;\textrm{REWARD}" />

  - *o*(**c**) is the number of contiguous ones starting in Position 1.
  - *z*(**c**) is the number of contiguous zeros ending in Position *D*.
  - if *o*(**c**) > *T* and *z*(**c**) > *T*, then REWARD is *D*, else REWARD is 0.
    - *T* is a user parameter.
- Deceptive-k Trap: There is a user parameter *k* which determines the number of dependencies of each variable.

  <img src="https://latex.codecogs.com/gif.latex?f(\boldsymbol{c})&space;=&space;\sum_{i=0}^{D/3-1}g(c_{3i&plus;1},c_{3i&plus;2},c_{3i&plus;3})," />
  <br>
  <img src="https://latex.codecogs.com/gif.latex?g(c_1,&space;c_2,&space;c_3)&space;=&space;\left\{&space;\begin{array}{ll}&space;1-d&space;&&space;\sum_{i}c_i&space;=&space;0&space;\\&space;1-2d&space;&&space;\sum_{i}c_i&space;=&space;1&space;\\&space;0&space;&&space;\sum_{i}c_i&space;=&space;2&space;\\&space;1&space;&&space;\sum_{i}c_i&space;=&space;3,&space;\\&space;\end{array}&space;\right." />

  - where *k* = 3 and *d* is a user parameter
- [NK-landscape](http://ncra.ucd.ie/wp-content/uploads/2020/08/SocialLearning_GECCO2019.pdf)
- [W-Model](http://iao.hfuu.edu.cn/images/publications/W2018TWMATBBDOBPIFTBGW.pdf)

## Usage
The example is shown below.  
In this example, the input `x` to the function is randomly generated.  
The variable `evals` indicates the evaluation value of each vector and the variable `info` includes information related to calculating the evaluation value for some problems such as Four-peaks function.

```
>>> import numpy as np
>>> from bbdob import OneMax
>>> from bbdob.utils import idx2one_hot
>>> dimension = 5
>>> objective = OneMax(dimension, minimize=True)
>>> x = np.random.randint(0, 2, (3, dimension))
>>> x
array([[0, 1, 1, 0, 1],
       [1, 0, 0, 0, 1],
       [0, 1, 0, 0, 0]])
>>> x = idx2one_hot(x, 2)
>>> x
array([[[1, 0],
        [0, 1],
        [0, 1],
        [1, 0],
        [0, 1]],

       [[0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1]],

       [[1, 0],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0]]])
>>> evals, info = objective(x)
>>> evals
array([-3, -2, -1])
>>> info
{}
```

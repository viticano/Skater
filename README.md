# PyInterpret

###Layout

![layout](https://raw.githubusercontent.com/datascienceinc/PyInterpret/master/PyInterpret.png)

### Dev Installation
```
git clone git@github.com:datascienceinc/PyInterpret.git
cd PyInterpret
sudo pip install -r requirements.txt
```

### Prod Installation
```
git clone git@github.com:datascienceinc/PyInterpret.git
cd PyInterpret
sudo python setup.py install
```


### Use this kind of this stuff to do cool stuff.

```
from PyInterpret.explanations import Explainer
import numpy as np
from scipy.stats import norm


#gen some data
B = np.array([1.8, -1.2, 3.1])
X = np.random.normal((1,2,3),(1,2,3), size=(1000, 3))
e = norm(0, 5)
y = np.dot(X, B) + e.rvs(1000)


#model it
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X, y)


#explain it
explainer = Explainer(X, rf.predict)
explainer.partial_dependence(0) #some great stuff happens here
explainer.explain_instance(X[0]) #and some other magic here

```


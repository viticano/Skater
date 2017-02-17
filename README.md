# PyInterpret

### Use this kind of this stuff to do cool stuff.

```

import PyInterpret
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
explainer = PyInterpret.Explainer(X)
explainer.partial_dependence(0)
explainer.explain_instance(X[0], rf.predict)


```

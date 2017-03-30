# PyInterpret
##### Master: ![Build Status-master](https://api.travis-ci.com/repositories/datascienceinc/model-interpretation.svg?token=okdWYn5kDgeoCPJZGPEz&branch=master)

<!--![layout](../master/PyInterpret.png?raw=true)
=======
-->

### Dev Installation
```
git clone git@github.com:datascienceinc/PyInterpret.git
cd PyInterpret
sudo python setup.py install
```

### Prod Installation (platform)
Requires that pip.conf is configured with pypi.datascience.com credentials.

```
pip install pyinterpret
```


### Use this kind of this stuff to do cool stuff.

```
import numpy as np
from scipy.stats import norm

#gen some data
B = np.random.normal(0, 10, size = 3)
X = np.random.normal(0,10, size=(1000, 3))
feature_names = ["feature_{}".format(i) for i in xrange(3)]
e = norm(0, 5)
y = np.dot(X, B) + e.rvs(1000)
example = X[0]

#model it
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X, y)


#partial dependence
from pyinterpret.core.explanations import Interpretation
i = Interpretation()
i.load_data(X, feature_names = feature_names)
i.partial_dependence.plot_partial_dependence([feature_names[0], feature_names[1]],
                                            regressor.predict)

#local interpretation
from pyinterpret.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X, feature_names = feature_names)
explainer.explain_instance(example,  regressor.predict).show_in_notebook()

```

### Testing
```
python pyinterpret/tests/all_tests.py --debug --n=1000 --dim=3 --seed=1
```

### API documentation
```
https://datascienceinc.github.io/model-interpretation/py-modindex.html
```

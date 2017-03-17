# PyInterpret
##### v1-dev: ![Build Status-master](https://api.travis-ci.com/repositories/datascienceinc/model-interpretation.svg?token=okdWYn5kDgeoCPJZGPEz&branch=v1-dev)

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
from pyinterpret.core.explanations import Interpretation
i = Interpretation()
i.load_data(regressor_X)
i.partial_dependence.plot_partial_dependence([feature_id1, feature_id2],regressor.predict)
```

### Testing
```
python pyinterpret/tests/test_all.py --debug --n=1000 --dim=3 --seed=1
```

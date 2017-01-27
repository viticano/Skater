# PyInterpret

### Use this kind of this stuff to do cool stuff.

M = LocalModel()
M.load_from_object(model.predict)

interpretation = Interpretation(M, data = X)
interpretation.partial_dependence()


interpretation.regress_model()

"""Base ModelInterpreter class. submodules like lime and partial dependence"
must have these methods"""


class ModelInterpreter(object):
    """
    Base Interpreter class. Common methods include loading a dataset and type setting.
    """

    def __init__(self, interpreter):
        self.interpreter = interpreter

    @property
    def data_set(self):
        """data_set routes to the Interpreter's dataset"""
        return self.interpreter.data_set

    @staticmethod
    def _types():
        return ['partial_dependence', 'lime']

    def load_data(self, training_data, index=None, feature_names=None):
        """.consider routes to Interpreter's .consider"""
        self.interpreter.consider(training_data, index=index, feature_names=feature_names)

    def build_annotated_model(self, prediction_function, target_names=None, examples=None):
        """
        returns skater.model.InMemoryModel
        Parameters
        ----------
            prediction_function(callable):
                the machine learning model "prediction" function to explain, such that
                predictions = predict_fn(data).

                For instance:
                    from sklearn.ensemble import RandomForestClassier
                    rf = RandomForestClassier()
                    rf.fit(X,y)
                    Interpreter.build_annotated_model(rf.predict)
            examples(np.ndarray):
                Examples to pass through the prediction_function to make inferences
                about what it outputs
        """
        return self.interpreter.build_annotated_model(prediction_function,
                                                      target_names=target_names,
                                                      examples=examples)

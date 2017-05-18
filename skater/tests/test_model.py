import unittest

import numpy as np
import pandas as pd
import subprocess
from multiprocessing import Process
import requests
import time
import sys
import tempfile
import os

from skater.core.explanations import Interpretation
from skater.util import exceptions
from arg_parser import create_parser
from skater.model import InMemoryModel, DeployedModel


class TestModel(unittest.TestCase):

    def setUp(self):
        args = create_parser().parse_args()
        debug = args.debug
        self.r_deploy_test = args.r_deploy_test
        self.python_deploy_test = args.python_deploy_test
        self.seed = args.seed
        self.n = args.n
        self.x1 = np.random.choice(range(10), replace=True, size=self.n)
        self.x2 = np.random.normal(0, 5, size=self.n)
        self.x3 = np.random.choice(['a','b','c'], replace=True, size=self.n)
        self.X = pd.DataFrame({'x1': self.x1, 'x2': self.x2, 'x3': self.x3}).values
        self.y = self.underlying_model_agg(self.X)
        self.tempdir = tempfile.mkdtemp()

    def test_in_memory_regressor(self):
        model = InMemoryModel(self.underlying_model_agg)
        example = np.array([2.0, 2.0, 'c'])
        assert model.predict(example) == np.array([1.2 -.7 * 2])

    def underlying_model(self, x1, x2, x3):
        x1 = float(x1)
        x2 = float(x2)
        if x3 == 'c' and x1 < 3 and x2 > 0:
            return 1.2 - .7 * x1
        elif x3 == 'c' and x1 < 6 and x2 > 0:
            return 1.2 + .7 * x1
        elif x3 == 'c' and x2 > 0:
            return 5.2 + 1.7 * x1
        elif x3 == 'c':
            return -.2 + 1.3 * x1
        elif x3 == 'b' and x1 < 3 and x2 > 0:
            return 11.2 - 2.7 * x1
        elif x3 == 'b' and x1 < 6 and x2 > 0:
            return 3.2 + 1.7 * x1
        elif x3 == 'b' and x2 > 0:
            return 5.2 - 1.7 * x1
        elif x3 == 'b':
            return 8.0 + 5.3 * x1
        else:
            return 3.7 * x1 * x2

    def underlying_model_agg(self, x):
        if len(x.shape)==2:
            results = []
            for row in x:
                x1, x2, x3 = row[0], row[1], row[2]
                results.append(self.underlying_model(x1, x2, x3))
            return np.array(results)
        else:
            x1, x2, x3 = x[0], x[1], x[2]
            result = self.underlying_model(x1, x2, x3)
            return np.array([result])


    def r_input_formatter(self, data):
        return {"input": pd.DataFrame(data).to_json(orient='records')}

    def r_output_formatter(self, response, key='probability'):
        return np.array(response.json()['probability'])

    def test_r_deploy_model(self):
        if self.r_deploy_test:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            feature_name = ['Status of existing checking account', 'Duration in month', 'Credit history'
                , 'Purpose', 'Credit amount', 'Savings account.bonds', 'Employment years'
                , 'Installment rate in percentage of disposable income'
                , 'Personal status and sex', 'Other debtors.guarantors', 'Present residence since'
                , 'Property', 'Age in years', 'Other installment plans', 'Housing',
                            'Number of existing credits at this bank'
                , 'Job', 'Number of people being liable to provide maintenance for', 'Telephone', 'Foreign worker',
                            'Status']

            p = self.R_async()
            p.join()
            with open(os.path.join(self.tempdir, 'run_r')) as f:
                while 'Starting server to listen on port 8000' not in f.read():
                    time.sleep(1)
                    print f.read()
                    print "waiting for server"

            f_n = [f.replace(' ', '.') for f in feature_name]
            input_data = pd.read_csv(url, sep=' ', names=f_n)
            selected_input_data = input_data[['Status.of.existing.checking.account', 'Duration.in.month', 'Credit.history',
                                              'Savings.account.bonds']]

            deployed_model_uri = "http://datsci.dev:8000/predict"
            dep_model = DeployedModel(deployed_model_uri,
                                      self.r_input_formatter,
                                      self.r_output_formatter,
                                      examples=selected_input_data.head(5))

            feature_names = np.array(selected_input_data.columns)
            interpreter = Interpretation(training_data=selected_input_data.head(5),
                                         feature_names=feature_names)

            plots = interpreter.partial_dependence.plot_partial_dependence(interpreter.data_set.feature_ids,
                                                                           dep_model,
                                                                           with_variance=True,
                                                                           sampling_strategy='random-choice',
                                                                           n_jobs=4,
                                                                           grid_resolution=10,
                                                                           n_samples=500,
                                                                           sample=True)

            #p.kill()
            tempfile.mkstemp()

    def sys_run_R_model(self):
        logfile = open(os.path.join(self.tempdir, 'r_run'), 'w')
        process = subprocess.Popen(["Rscript", "util/deployme.R"], stdout=logfile)#, stdin=None, stdout=None, stderr=None)
        return process, logfile

    # def wrap(self, task, name):
    #     def wrapper(*args, **kwargs):
    #         with open(os.path.join(self.tempdir, name), 'w') as f:
    #             task(*args, **kwargs)
    #             sys.stdout = f
    #
    #     return wrapper

    def R_async(self):
        #p = Process(target=self.wrap(self.sys_run_R_model, 'run_r'))
        p = Process(target=self.sys_run_R_model)
        #p.daemon = True
        p.start()
        return p
        #return self.sys_run_R_model()


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(suite)

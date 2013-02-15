#! -*- coding:utf-8 -*-

import theano
import pylearn2
from pylearn2.config import yaml_parse
from pylearn2.datasets import preprocessing
import contest_dataset

dataset = """!obj:contest_dataset.ContestDataset {
          which_set: 'train',
          start: 0,
          stop: 3500
     }"""

model = """!obj:pylearn2.models.mlp.MLP {
        layers: [ !obj:pylearn2.models.mlp.SoftmaxPool {
                         layer_name: 'h0',
                         detector_layer_dim: 1200,
                         pool_size: 1,
                         sparse_init: 15,
                     }, !obj:pylearn2.models.mlp.Softmax {
                         layer_name: 'y',
                         n_classes: 7,
                         irange: 0.
                     }
                    ],
              nvis: 2304,
        }"""

algorithm = """!obj:pylearn2.training_algorithms.sgd.SGD {
            batch_size: 100,
            learning_rate: 0.0004,
            init_momentum: .99,
            monitoring_dataset:
              {
                 'train' : *train,
                 'valid' : !obj:contest_dataset.ContestDataset {
                               which_set: 'train',
                               start: 3500,
                               stop: 4178
                           }
              },
            cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [ 
                        !obj:pylearn2.costs.cost.MethodCost {
                             method: 'cost_from_X', 
                             supervised: 1
                        }, !obj:pylearn2.models.mlp.WeightDecay {
                           coeffs: [ .00005, .00005 ]
                        }
                        ]
            },
            termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
               channel_name: 'valid_y_misclass',
               prop_decrease: 0.,
               N: 10
            }
     }"""

train = """!obj:pylearn2.train.Train {
        dataset: &train %(dataset)s,
        model: %(model)s,
        algorithm: %(algorithm)s,
        extensions: [ 
           !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                channel_name: 'valid_y_misclass',
                save_path: "mlp_best.pkl"
           },
           !obj:pylearn2.training_algorithms.sgd.OneOverEpoch {
                start: 1
           }
       ],
       save_path: "mlp.pkl",
       save_freq: 1
}""" % locals()

train = yaml_parse.load(train)
train.main_loop()

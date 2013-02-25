#! -*- coding:utf-8 -*-

import theano
import pylearn2
from pylearn2.config import yaml_parse
from pylearn2.datasets import preprocessing
import contest_dataset

dataset = """!obj:contest_dataset.ContestDataset {
          which_set: 'train',
          start: 0,
          stop: 3500,
          preprocessor : !obj:pylearn2.datasets.preprocessing.GlobalContrastNormalization { }
     }"""

model = """!obj:pylearn2.models.mlp.MLP {
        batch_size: 100,
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [48, 48],
            num_channels: 1
        },
        layers: [ !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h2',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.ConvRectifiedLinear {
                     layer_name: 'h3',
                     output_channels: 64,
                     irange: .05,
                     kernel_shape: [5, 5],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     max_kernel_norm: 1.9365
                 }, !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 7,
                     istdev: .05
                 }
                ],
    }"""

algorithm = """ !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .001,
        init_momentum: .5,
        monitoring_dataset:
            {
                 'train' : *train,
                 'valid' : !obj:contest_dataset.ContestDataset {
                               which_set: 'train',
                               start: 3500,
                               stop: 4178,
                               preprocessor : !obj:pylearn2.datasets.preprocessing.GlobalContrastNormalization { }  
                           }
            },
        cost: !obj:pylearn2.costs.cost.SumOfCosts { costs: [
            !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
            }, !obj:pylearn2.models.mlp.WeightDecay {
                coeffs: [ .00005, .00005, .00005 ]
            }
            ]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass",
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
                save_path: "conv_net_best.pkl"
           },
           !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                start: 1,
                saturate: 10,
                final_momentum: .99
        }
       ],
       save_path: "conv_net.pkl",
       save_freq: 1
}""" % locals()

train = yaml_parse.load(train)
train.main_loop()

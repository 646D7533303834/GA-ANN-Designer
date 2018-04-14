# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import random

from tensorflow.examples.tutorials.mnist import input_data
from deap import creator, base, tools, algorithms

import tensorflow as tf

FLAGS = None

POP_SIZE = 10
SELECTION_SIZE = int(POP_SIZE * 0.2)

NUM_OUTPUTS = 10

BOUND_LOW = [1, 1, 1, 1]
BOUND_UP = [11, 11, 10, 1024]

BOUND_LAYER_LOW = 1
BOUND_LAYER_HIGH = 10

BOUND_NODES_LOW = 1
BOUND_NODES_HIGH = 1024

CXPB, MUTPB = 0.25, 0.625

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                    help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()

# Import data
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
#
toolbox.register("attr_act", random.randint, 1, 11)
toolbox.register("attr_opt", random.randint, 1, 11)
toolbox.register("attr_num_layer", random.randint, BOUND_LAYER_LOW, BOUND_LAYER_HIGH)
toolbox.register("attr_num_nodes", random.randint, BOUND_NODES_LOW, BOUND_NODES_HIGH)
# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of an activation function,
#                         optimizer function, loss function,
#                         number of hidden layers, and number of
#                         nodes per layer
toolbox.register("individual",
                 tools.initCycle,
                 creator.Individual,
                 (toolbox.attr_act,
                  toolbox.attr_opt,
                  toolbox.attr_num_layer,
                  toolbox.attr_num_nodes
                 ),
                 n=1)
# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def mutateFunction(individual):
    if random.random() < 0.5:
        individual[2] = random.randint(BOUND_LAYER_LOW, BOUND_LAYER_HIGH)
    else:
        individual[3] = random.randint(BOUND_NODES_LOW, BOUND_NODES_HIGH)


def createModel(inputLayer, actFunc, layers, nodes, init=True):
    if init == True:
        print("init")
        return tf.layers.dense(inputs=createModel(inputLayer, actFunc, layers-1, nodes, init=False), units=NUM_OUTPUTS,
                               activation=actFunc, use_bias=True)
    elif layers <= 1:
        print("input layer")
        return inputLayer
    else:
        print("hidden layer: ", layers)
        return tf.layers.dense(inputs=createModel(inputLayer, actFunc, layers-1, nodes, init=False), units=nodes,
                               activation=actFunc, use_bias=True)

def getActivationFunction(index, input, weight, bias):
    return tf.nn.relu_layer(input, weight, bias)


def createMulModel(inLayer, activation_function, layers, nodes, init=True):
    """if ((init==True) and (layers==1)):
        wa = tf.Variable(tf.zeros([784, NUM_OUTPUTS]))
        ba = tf.Variable(tf.zeros([NUM_OUTPUTS]))
        return tf.matmul(inLayer, wa) + ba
    el"""
    # if init==True:
    #     if layers > 1:
    #         return tf.layers.dense(
    #             inputs=createMulModel(inLayer, activation_function, (layers - 1), nodes),
    #             units=NUM_OUTPUTS,
    #             activation=tf.nn.relu
    #         )
    #     else:
    #         return tf.layers.dense(inputs=inLayer, units=NUM_OUTPUTS)
    if layers <= 1:
        return tf.layers.dense(inputs=inLayer, units=nodes, activation=tf.nn.relu)
    else:
        return tf.layers.dense(
            inputs=createMulModel(inLayer, activation_function, (layers-1), nodes),
            units=nodes,
            activation=tf.nn.relu
        )


def evalModel(individual):
    startTime = time.time()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    #W = tf.Variable(tf.zeros([784, 10]))
    #b = tf.Variable(tf.zeros([10]))
    #y = tf.matmul(x, W) + b
    y = createMulModel(x, individual[0], individual[2], individual[3])

    logits = tf.layers.dense(inputs=y, units=NUM_OUTPUTS)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_OUTPUTS])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    i = 0
    while i < 1000: # and (time.time() - startTime) < 60:
        i = i + 1
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images,
    #                              y_: mnist.test.labels}))
    result = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                  y_: mnist.test.labels})
    elapsedTime = time.time() - startTime
    #print('Time: ', elapsedTime)

    return (result,)


toolbox.register("evaluate", evalModel)

toolbox.register("mate", tools.cxUniform, indpb=0.5)

toolbox.register("mutate", mutateFunction)

toolbox.register("select", tools.selTournament, tournsize=2)

pop = toolbox.population(n=POP_SIZE)

print("Start of evolution")

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
    print("ind.fitness.values: ", ind.fitness.values)

fits = [ind.fitness.values[0] for ind in pop]

g = 0
start_time = time.time()
time_elapsed = 0

while max(fits) < 0.99999 and g < 1000 and time_elapsed < 3600:

    g = g + 1
    print("-- Generation %i --" % g)

    top = tools.selBest(pop, SELECTION_SIZE)

    top = list(map(toolbox.clone, top))

    offspring = toolbox.select(pop, (POP_SIZE-SELECTION_SIZE))

    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(child1, child2)

        del child1.fitness.values
        del child2.fitness.values

    for mutant in offspring[2::]:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    offspring.extend(top)

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    best_ind = tools.selBest(pop,1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    # layer_mean = sum(pop[::, 2])/length
    # node_mean = sum(pop[::, 3])/length
    # print("  Avg Layers: %s" % layer_mean)
    # print("  Avg Nodes:  %s" % node_mean)

    time_elapsed = time.time() - start_time

print("-- End of evolution --")

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

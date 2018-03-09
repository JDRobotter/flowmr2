#

import tensorflow as tf
import numpy as np
import os,sys,glob
import logging
#logging.getLogger().setLevel(logging.INFO)

def cnn_model_fn(features, labels, mode):
    
    input_layer = tf.reshape(features["x"], [-1, 40*3, 30, 1])

    # convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=8,
        kernel_size=[3,1],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,1], strides=(3,1))

    # convolutionnal layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=8,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 20*15*8])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training= mode == tf.estimator.ModeKeys.TRAIN)

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=4)

    predictions = {
        'classes': logits,
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculate loss
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    # configure the traing op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step= tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def walk_concatenate(path,op):
    ls = []
    for fname in glob.glob(path):
        v = np.frombuffer(open(fname,'rb').read(), dtype=np.uint8)
        ls.append(op(v))
    return np.array(ls,dtype=np.float16)

def train(directory):
    print("[+] loading",directory)
    vimgs = walk_concatenate(
        os.path.join(directory,"img_*"),
        # transform 8bit color channels to 0-1 floats
        np.vectorize(lambda x: x/255.0))

    vkbs = walk_concatenate(
        os.path.join(directory,"kb_*"),
        # transform ascii chars to 0-1 values
        np.vectorize(lambda x: 0 if x==48 else 1))

    print(" -> ",len(vimgs),len(vkbs),"loaded")
    return vimgs,vkbs

def main():
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model")

    logging_hook = tf.train.LoggingTensorHook(
        tensors={"probabilities":"softmax_tensor"},
        every_n_iter=1)

    vimgs,vkbs = train("session_2018_03_06__21_03_54")
    
    N = 100
    vimgs_eval = vimgs[:N]
    vimgs_train = vimgs[N:]
    vkbs_eval = vkbs[:N]
    vkbs_train = vkbs[N:]

    while True:
        if 1:
            print("[+] training")
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x":vimgs_train},
                y=vkbs_train,
                batch_size=100,
                num_epochs=None,
                shuffle=True)

            classifier.train(
                input_fn=train_input_fn,
                steps=50,
                hooks=[logging_hook])

        if 1:
            print("[+] evaluating")
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x":vimgs_eval},
                y=vkbs_eval,
                num_epochs=1,
                shuffle=False)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)


if __name__ == '__main__':
    main()

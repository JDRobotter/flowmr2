#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os,sys,glob
import logging
logging.getLogger().setLevel(logging.INFO)

import loader


def model(training=False):

    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
    # inputs
    x = tf.placeholder(dtype=tf.float16, shape=(None, 30, 40, 3))
    # onehots label
    y = tf.placeholder(dtype=tf.uint8, shape=(None, 4))

    # convolutional layer #1 (will apply a conv layer on RGB values)
    conv1 = tf.layers.conv2d(
                inputs=x,
                filters=8,
                kernel_size=[1,1],
                padding="same",
                activation=tf.nn.relu)

    # convolutional layer #2 (spatial)
    conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=16,
                kernel_size=[5,5],
                padding="same",
                activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # dense layer
    pool2_flat = tf.reshape(pool2, (-1, 15*20*16))
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=training)

    # logits layer
    logits = tf.layers.dense(inputs=dropout, units=4)

    # probabilities
    probabilities = tf.nn.softmax(logits, name="softmax_tensor")

    # loss function
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
    tf.summary.scalar("loss",loss)

    # optimizer
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(
                loss, global_step=global_step_tensor)

    # convert logits to prediction
    shape = tf.shape(probabilities)
    prediction = tf.where(
                    tf.less(probabilities, 0.6*tf.ones(shape, np.float16)),
                    tf.zeros(shape, np.float16),
                    tf.ones(shape, np.float16))

    accuracy = tf.metrics.accuracy(y, prediction)
    tf.summary.scalar("accuracy",accuracy[1])

    return global_step_tensor,x,y,train_op,loss,logits,prediction,accuracy

def main():
    
    vimgs,vkbs = loader.load_session("session_2018_03_06__21_03_54")
    vimgs = np.reshape(vimgs,(-1,30,40,3))
    vkbs = np.reshape(vkbs,(-1,4))

    global_step,x,y,train,loss,logits,prediction,accuracy = model()
    with tf.Session() as s:

        writer = tf.summary.FileWriter("./model/", s.graph)
        merged = tf.summary.merge_all()

        # restore graph
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint("./model")

        s.run(tf.local_variables_initializer())
        if ckpt is None:
            s.run(tf.global_variables_initializer())

        else:
            saver.restore(s, ckpt)

        # training loop
        batch_size = 100
        while True:
            step = tf.train.global_step(s, global_step)

            i = batch_size*step % len(vimgs)
            j = batch_size*(step+1) % len(vimgs)
            vimgs_batch = vimgs[i:j]
            vkbs_batch = vkbs[i:j]
            summary,oy,_,ologits,op,oloss,oa = s.run(
                    [merged,y,train,logits,prediction,loss,accuracy],
                    feed_dict={x:vimgs_batch,y:vkbs_batch})
            print(step)
            print(zip(op,oy))
            print(oloss,oa)

            saver.save(s, "./model/model.ckpt", global_step=step)
            writer.add_summary(summary,step)


if __name__ == '__main__':
    main()

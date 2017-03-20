from nn import add_layer
import pickle
import numpy as np
import tensorflow as tf
from DataSet import DataSet
from sklearn import preprocessing

hidden_cell = 33

with open('../dataset/train_X.pkl','rb') as f:
    train_X = pickle.load(f)
with open('../dataset/train_y.pkl','rb') as f:
    train_y = pickle.load(f)
with open('../dataset/test_X.pkl','rb') as f:
    test_X = pickle.load(f)
with open('../dataset/test_y.pkl','rb') as f:
    test_y = pickle.load(f)

# train_X = preprocessing.scale(train_X)
train_Set = DataSet(train_X, train_y)

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 6])
    ys = tf.placeholder(tf.float32, [None, 2])

hidden = add_layer(xs, len(train_X[0]), hidden_cell, layer_name='hidden', keep_prob=1., activation_function=tf.nn.tanh)
logits = add_layer(hidden, hidden_cell, len(train_y[0]), layer_name='logits', keep_prob=1., activation_function = None)

with tf.name_scope('losses'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys, name='cross_entropy'))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    SGD_train = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    Adam_train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    

with tf.name_scope('accuray'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(ys, 1))
    accuary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuary', accuary)

saver = tf.train.Saver()
sess = tf.Session()
merged = tf.summary.merge_all()
writer_train = tf.summary.FileWriter('logs/w1', sess.graph)
writer_test = tf.summary.FileWriter('logs/w2', sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

for i in range(3000):
    data_batch, label_batch = train_Set.next_batch(256)
    sess.run(Adam_train, feed_dict={xs:data_batch, ys:label_batch})
    if i%100 == 0:
        result_accuray = sess.run(accuary, feed_dict={xs:test_X, ys:test_y})
        print(result_accuray)
        train_result = sess.run(merged, feed_dict={xs:data_batch, ys:label_batch})
        test_result = sess.run(merged, feed_dict={xs:test_X, ys:test_y})
        writer_train.add_summary(train_result, i)
        writer_test.add_summary(test_result, i)

saver.save(sess, '../model/es_nn_model.ckpt')
    
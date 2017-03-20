from nn import add_layer
import pickle
import numpy as np
import tensorflow as tf

with open('../dataset/test_data.pkl','rb') as f:
    test_set = pickle.load(f, encoding="latin1")
test_set = np.array(test_set, dtype=np.float32)
test_X, test_y = test_set[:,:6], test_set[:,-1:]
test_y = test_y.flatten()
one_hot = tf.one_hot(test_y, depth=2, on_value=1., off_value=0.)
with tf.Session() as sess:
    test_y = sess.run(one_hot)

with open('../dataset/test_X.pkl', 'wb') as f:
    pickle.dump(test_X, f)

with open('../dataset/test_y.pkl', 'wb') as f:
    pickle.dump(test_y, f)








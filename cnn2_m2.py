import tensorflow as tf
import numpy as np

###
class TextCNN_M2(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
      self, post_length, art_length, tags_length, num_classes,
      embedding_size, filter_sizes_w, filter_sizes_t, filter_sizes_a, num_filters, batch_size, embeddings_w, embeddings_t, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_t = tf.placeholder(tf.int32, [None, tags_length], name="input_t")
        self.input_a = tf.placeholder(tf.int32, [None, art_length], name="input_a")
        self.input_x = tf.placeholder(tf.int32, [None, post_length], name="input_x")
        self.input_y = tf.placeholder(tf.bool, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("word_embeddings_t"):
            self.W_t = tf.to_float(embeddings_t, name='tag_embeddings')
            self.embedded_chars_t = tf.nn.embedding_lookup(self.W_t, self.input_t)
            self.embedded_chars_expanded_t = tf.expand_dims(self.embedded_chars_t, -1)
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("word_embeddings_w"):
            self.W_w = tf.to_float(embeddings_w, name='post_embeddings')
            self.embedded_chars_w = tf.nn.embedding_lookup(self.W_w, self.input_x)
            self.embedded_chars_expanded_w = tf.expand_dims(self.embedded_chars_w, -1)
        with tf.device('/cpu:0'), tf.name_scope("word_embeddings_a"):
            self.embedded_chars_a = tf.nn.embedding_lookup(self.W_w, self.input_a)
            self.embedded_chars_expanded_a = tf.expand_dims(self.embedded_chars_a, -1)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_w = []
        for i, filter_size_w in enumerate(filter_sizes_w):
            with tf.name_scope("conv-maxpool-w-%s" % filter_size_w):
                # Convolution Layer
                filter_shape_w = [filter_size_w, embedding_size, 1, num_filters]
                W_w = tf.Variable(tf.truncated_normal(filter_shape_w, stddev=0.1), name="W_w")
                b_w = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_w")
                conv_w = tf.nn.conv2d(
                    self.embedded_chars_expanded_w,
                    W_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_w")
                # Apply nonlinearity
                h_w = tf.nn.relu(tf.nn.bias_add(conv_w, b_w), name="relu_w")
                # Maxpooling over the outputs
                pooled_w = tf.nn.max_pool(
                    h_w,
                    ksize=[1, post_length - filter_size_w + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_w")
                pooled_outputs_w.append(pooled_w)

        # Combine all the pooled features
        num_filters_total_w = num_filters * len(filter_sizes_w)
        self.h_pool_w = tf.concat(pooled_outputs_w,3)
        self.h_pool_flat_w = tf.reshape(self.h_pool_w, [-1, num_filters_total_w])

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_t = []
        for i, filter_size_t in enumerate(filter_sizes_t):
            with tf.name_scope("conv-maxpool-t-%s" % filter_size_t):
                # Convolution Layer
                filter_shape_t = [filter_size_t, embedding_size, 1, num_filters]
                W_t = tf.Variable(tf.truncated_normal(filter_shape_t, stddev=0.1), name="W_t")
                b_t = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_t")
                conv_t = tf.nn.conv2d(
                    self.embedded_chars_expanded_t,
                    W_t,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_t")
                # Apply nonlinearity
                h_t = tf.nn.relu(tf.nn.bias_add(conv_t, b_t), name="relu_t")
                # Maxpooling over the outputs
                pooled_t = tf.nn.max_pool(
                    h_t,
                    ksize=[1, tags_length - filter_size_t + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_t")
                pooled_outputs_t.append(pooled_t)

        # Combine all the pooled features
        num_filters_total_t = num_filters * len(filter_sizes_t)
        self.h_pool_t = tf.concat(pooled_outputs_t,3)
        self.h_pool_flat_t = tf.reshape(self.h_pool_t, [-1, num_filters_total_t])


       # Create a convolution + maxpool layer for each filter size
        pooled_outputs_a = []
        for _ in xrange(2):
            for i, filter_size_a in enumerate(filter_sizes_a):
                with tf.name_scope("conv-maxpool-a-%s" % filter_size_a):
                    # Convolution Layer
                    filter_shape_a = [filter_size_a, embedding_size, 1, num_filters]
                    W_a = tf.Variable(tf.truncated_normal(filter_shape_a, stddev=0.1), name="W_a")
                    b_a = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_a")
                    conv_a = tf.nn.conv2d(
                        self.embedded_chars_expanded_a,
                        W_a,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv_a")
                    # Apply nonlinearity
                    h_a = tf.nn.relu(tf.nn.bias_add(conv_a, b_a), name="relu_a")
                    # Maxpooling over the outputs
                    pooled_a = tf.nn.max_pool(
                        h_a,
                        ksize=[1, art_length - filter_size_a + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool_a")
                    pooled_outputs_a.append(pooled_a)

        # Combine all the pooled features
        num_filters_total_a =2* num_filters * len(filter_sizes_a)
        self.h_pool_a = tf.concat(pooled_outputs_a,3)
        self.h_pool_flat_a = tf.reshape(self.h_pool_a, [-1,num_filters_total_a])
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(tf.concat([self.h_pool_flat_t,self.h_pool_flat_w],1), self.dropout_keep_prob)

        with tf.name_scope('Multiplied_filters'):
            self.mul_layer=tf.multiply(self.h_drop,self.h_pool_flat_a,name='Mul_layer')

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total_w+num_filters_total_t, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores =  tf.nn.sigmoid(tf.nn.xw_plus_b(self.mul_layer, W, b, name="scores"))
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = -tf.reduce_mean(((tf.to_float(self.input_y) * tf.log(self.scores+.00001)    ) + ((1 - tf.to_float(self.input_y)) * tf.log(1.00001 - self.scores))))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(tf.cast(self.input_y,"float16"), 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float16"), name="accuracy")

        with tf.name_scope('F1'):
            argmax_prediction = self.predictions
            argmax_y = tf.argmax(tf.cast(self.input_y, 'float16'), 1)

            TN = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
            TP = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
            FN = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)	
            FP = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
            denom=2*TP+FP+FN
            self.f1=2*TP/(denom+1e-8)
            '''
            ra=tf.range(tf.divide(tf.size(self.input_y),tf.constant(100)))
            rb=tf.range(tf.divide(tf.size(self.input_y),tf.constant(100)))
            rc=tf.range(tf.divide(tf.size(self.input_y),tf.constant(100)))
            r=tf.to_int32(tf.reshape(tf.stack([ra,rb,rc],axis=1),[-1]))
            c=tf.reshape(self.top3,[-1])
            cordinates = tf.transpose(tf.stack([r,c],axis=0))
            top3ar = tf.sparse_to_dense(cordinates, tf.shape(self.input_y), 1, validate_indices=False)
            top3armul=tf.multiply(top3ar,tf.to_int32(self.input_y))
            _,t3y=tf.nn.top_k(tf.to_int32(self.input_y),k=3)
            t3yr=tf.reshape(t3y,[-1])
            cors=tf.transpose(tf.stack([r,t3yr],axis=0))
            t3yar=tf.sparse_to_dense(cors, tf.shape(self.input_y), 1, validate_indices=False)
            t3ymul=tf.multiply(t3yar,tf.to_int32(self.input_y))
            maxac=tf.reduce_sum(t3ymul)
            preac=tf.reduce_sum(top3armul)
            #self.acc_tool=acc_tool
            #correct_predictions = tf.gather(tf.reshape(self.input_y,[-1]),tf.add(self.acc_tool,self.predictions))
            self.accuracy = tf.divide(preac,maxac, name="top_3_accuracy")
            #tf.reduce_mean(correct_predictions, name="accuracy")
            '''
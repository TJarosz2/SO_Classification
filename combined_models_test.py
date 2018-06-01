import tensorflow as tf
import numpy as np
import gensim
import nntextcleaner as tc
from cnn2_m2 import TextCNN_M2
from cnn_predict_domain_tags import TextCNN
eps=1e-7
tags=np.load('tags_we_500mc_words.npy')
words=np.load('dic_w2n_mc500.npy').item()
embeds_w=np.load('we_500mc_arrays.npy')
embeds_t=np.load('tags_we_500mc_arrays.npy')

m1_graph=tf.Graph()
with m1_graph.as_default():
		cnn_m1 = TextCNN(
            sequence_length=100,
            num_classes=100,
            tags_length=5,
            embedding_size=256,
            filter_sizes_w=list(map(int, "1,2,3,4,5".split(","))),
            filter_sizes_t=list(map(int, "1,2,3,4,5".split(","))),
            num_filters=500,
            batch_size=21,
            embeddings_w=embeds_w,
            embeddings_t=embeds_t,
            l2_reg_lambda=0)
		m1_saver = tf.train.Saver(tf.global_variables())

m1_sess=tf.Session(graph=m1_graph)


with m1_sess.as_default():
	with m1_graph.as_default():
		m1_sess.run(tf.global_variables_initializer())
		m1_saver.restore(m1_sess, "/Users/tommyjarosz/Desktop/research_cnn/Stack-Overflow-Analysis-Research/aws_results/runs/1510008839/checkpoints/model-96000") 
#m1_sess.close()

m2_graph=tf.Graph()
with m2_graph.as_default():
		cnn_m2 = TextCNN_M2(
            post_length=100,
            num_classes=2,
            tags_length=5,
            art_length=1024,
            embedding_size=256,
            filter_sizes_w=list(map(int, "1,2,3,4,5".split(","))),
            filter_sizes_t=list(map(int, "1,2,3,4,5".split(","))),
            filter_sizes_a=list(map(int, "1,2,3,4,5".split(","))),            
            num_filters=350,
            batch_size=75,
            embeddings_w=embeds_w,
            embeddings_t=embeds_t,
            l2_reg_lambda=0)
		m2_saver = tf.train.Saver(tf.all_variables())



m2_sess=tf.Session(graph=m2_graph)
with m2_sess.as_default():
	with m2_graph.as_default():
		m2_sess.run(tf.global_variables_initializer())
		m2_saver.restore(m2_sess, "/Users/tommyjarosz/Desktop/research_cnn/Stack-Overflow-Analysis-Research/aws_results_m2/runs/1510153563/checkpoints/model-46000")
#m2_sess.close()

while 1:
	post=raw_input('Enter a post, or "quit" to exit.')
	if post=='quit':
		break
	tag=raw_input('Enter Tags')
	tag=tag.decode('utf-8').strip()
	post_in_data=[]
	tag_in_data=[]
	for word in tc.clean_str_no_l(post.decode('utf-8').strip()):
		try:
			post_in_data.append(words[word])
		except:
			continue
		if len(post_in_data)>99:
			break
	while len(post_in_data)<100:
		post_in_data.append(0)
	for t in tag.split():
		try:
			tag_in_data.append(np.where(tags==t)[0][0])
		except:
			continue
		if len(tag_in_data)>4:
			break
	while len(tag_in_data)<5:
		tag_in_data.append(0)

	with m1_sess.as_default():
		with m1_graph.as_default():
			m1_feed_dict = {
	          cnn_m1.input_x: [post_in_data],
	          cnn_m1.input_t: [tag_in_data],
	          cnn_m1.dropout_keep_prob: 1.0,
	        }
			m1_out=m1_sess.run([cnn_m1.scores],m1_feed_dict)[0][0]
	#m1_sess.close()

	m1_out_sorted=list(reversed(sorted(range(len(m1_out)), key=lambda k: m1_out[k])))
	results=[]
	for domnumber in m1_out_sorted[0:10]:
		try:
			domdic=np.load('./Doms/dom'+str(domnumber)+'/dic.npy').item()
		except:
			try:
				domdic=np.load('./Doms/dom'+str(domnumber)+'/all_links.npy')
				for link in domdic:
					results.append(link[0])
				continue
			except:
				continue
		keys=domdic.keys()
		scores=[]
		ms=min(3,len(keys))
		for key in keys:
			m2_in_data=domdic[key]
			with m2_sess.as_default():
				with m2_graph.as_default():
					m2_feed_dict = {
		              cnn_m2.input_x: [post_in_data],
		              cnn_m2.input_t: [tag_in_data],
		              cnn_m2.input_a: [m2_in_data],
		              cnn_m2.dropout_keep_prob: 1.0,
		            }
					m2_out=m2_sess.run([cnn_m2.scores],m2_feed_dict)[0][0]
			#m2_sess.close()

			score=m2_out[1]/(m2_out[0]+eps)
			scores.append(score)

		sortedkeysindx=list(reversed(sorted(range(len(scores)), key=lambda k: scores[k])))

		sortedkeys=[keys[i] for i in sortedkeysindx]

		results.append(sortedkeys[0:ms])
	print "Check out these links!"
	print results

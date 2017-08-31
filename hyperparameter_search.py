from shared_key import SharedKeyCrypto, FLAGS
import tensorflow as tf
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import random

hyper_keys = ['ab_diversity', 'ab_eve1', 'ab_eve2', 'ab_decay', 'ab_learn',
              'eve_decay', 'eve_learn']

#Best run so far, at KEY = 8 / BATCH = 3000 / TRAINITER=ATTITER=2000 / AB_DEPTH  = 3 / EVE_DEPTH = 1
#score = -1.751314 at new 
#{'ab_diversity': 0.14400000000000002, 'eve_learn': 0.00013458798574153815, 'eve_decay': 0.46005119909369674, 'ab_eve2': 1.2, 'ab_learn': 0.00432, 'ab_eve1': 0.558163294467307, 'ab_decay': 0.0}
#(#s = 306)

def run_hyper(model, init, sess, hyper_dict, n=1):
  #If requesting several runs, median them together
  if n>1:
    results = []
    results += [run_hyper(model, init, sess, hyper_dict, n=1)]
    results = zip(*results)
    return list(map(np.median, results))

  feed_dict = {}
  for key in hyper_keys:
    feed_dict[model.__getattribute__(key)] = hyper_dict[key]

  #Each test run we need to re-randomize variables
  sess.run(init)
  for j in xrange(0,FLAGS.train_iter):    
    if j%1000==-1:
      print("j=%d" % j,
       sess.run([model.alice_bob_err, model.eve_err_1, model.eve_err_2],
                  feed_dict=feed_dict))
    sess.run(model.train_AB, feed_dict=feed_dict)
    sess.run(model.train_E, feed_dict=feed_dict)

  for j in xrange(0,FLAGS.attack_iter):    
    if j%100==-1:
      print("j=%d" % j,
       sess.run([model.alice_bob_err, model.eve_err_1, model.eve_err_2],
                  feed_dict=feed_dict))
    sess.run(model.train_E, feed_dict=feed_dict)

  res = sess.run([model.alice_bob_err, model.eve_err_1, model.eve_err_2],
                  feed_dict=feed_dict)
  print("Res = ", res)
  return res

#Positive score means Alice/Bob lose -- Eve cracked them.
#Consider using a softmax for the two Eve errors?
def calc_score(test_results):
  ABerr, Eerr1, Eerr2 = test_results
  return ABerr - min(Eerr1, Eerr2)

def search_hyper():
  #Initialize a list of hyperparameters to search across
  hyper_dict = {}
  for key in hyper_keys:
    hyper_dict[key] = FLAGS.__getattr__(key)

  #Instantiate model
  model = SharedKeyCrypto()
  init = tf.global_variables_initializer()  
  
  sample_size = 5

  with tf.Session() as sess:
    results = run_hyper(model, init, sess, hyper_dict, n=sample_size)
    score = calc_score(results)
    print("Score = %f at " % score, hyper_dict)
    
    while True:
      flip_var = random.choice(hyper_keys)
      is_ab_var = (flip_var[0:2] == 'ab')
      old_value = hyper_dict[flip_var]
      new_value = old_value * random.choice([1.2, 1/1.2])
      hyper_dict[flip_var] = new_value

      new_results = run_hyper(model, init, sess, hyper_dict, n=sample_size)
      new_score = calc_score(new_results)
      print("FLIPVAR = ",flip_var)

      is_better = is_ab_var ^ (score < new_score)
      if is_better:
        #leave new_value in hyper_dict
        score = new_score
        sample_size += 1
        print("Improved score = %f at new " % score, hyper_dict, "(#s = %d)" % sample_size)
      else:
        print("Worse score = %f at same " % new_score, hyper_dict)
        hyper_dict[flip_var] = old_value
        if random.uniform(0,1) > 0.6:
          fresh_results = run_hyper(model, init, sess, hyper_dict, n=sample_size)
          score = calc_score(fresh_results)      
          print("Fresh score = %f at same " % score, hyper_dict)
        if random.uniform(0,1) > 0.8:
          sample_size -= 1

def main(unused_argv):
  search_hyper()

if __name__ == '__main__':
  tf.app.run()

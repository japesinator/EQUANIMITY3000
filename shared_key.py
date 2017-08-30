import signal
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import math
import time

flags = tf.app.flags

flags.DEFINE_integer('key_size', 8, 'Key size')

flags.DEFINE_integer('batch_size', 3000, 'Batch size')

flags.DEFINE_integer('obfuscation_depth', 3, 'Depth to compute public key')
flags.DEFINE_integer('construction_depth', 3, 'Depth to compute shared key')

flags.DEFINE_integer('eve_depth', 1, 'Depth for Eve to attack')

flags.DEFINE_float('ab_diversity', 0.1, 'Alice/Bob key diversity penalty')
flags.DEFINE_float('ab_eve1', 2.0, 'Alice/Bob penalty for Eve guessing')
flags.DEFINE_float('ab_eve2', 1.0, 'Alice/Bob penalty for Eve guessing')
flags.DEFINE_float('ab_decay', 0.0, 'Alice/Bob weight decay')
flags.DEFINE_float('ab_learn', 0.003, 'Alice/Bob learning rate')

flags.DEFINE_float('eve_decay', 0.01, 'Eve weight decay')
flags.DEFINE_float('eve_learn', 0.001, 'Eve learning rate')


FLAGS = flags.FLAGS

KEY_SIZE = FLAGS.key_size

BATCH = FLAGS.batch_size

OBFUSCATION_DEPTH = FLAGS.obfuscation_depth
CONSTRUCTION_DEPTH = FLAGS.construction_depth
EVE_DEPTH = FLAGS.eve_depth

OBFUSCATION_MEMORY = OBFUSCATION_DEPTH
CONSTRUCTION_MEMORY = CONSTRUCTION_DEPTH
EVE_MEMORY = EVE_DEPTH



def get_rand_bits(batch_size, n):
  as_int = tf.random_uniform(
      [batch_size, n], minval=0, maxval=2, dtype=tf.int32)
  expanded_range = (as_int * 2) - 1
  return tf.cast(expanded_range, tf.float32)

class SharedKeyCrypto():
  
  #Generic "processing" function. Deep relu net with skip-conns and final tanh.
  def process(self, input_list, output_size, depth, memory):
    def fc_with_input(data, count):
      with tf.variable_scope('fc_'+count):
        stacked = tf.concat(input_list + data, axis=1)
        return tf.contrib.layers.fully_connected(stacked, output_size,
                  activation_fn = tf.nn.relu,
                  biases_initializer = tf.random_normal_initializer)
    data = []
    for i in xrange(depth):
      data.append(fc_with_input(data, str(i)))
      if i >= memory:
        data = data[1:] #Trim off the working space once it becomes too big

    stacked = tf.concat(input_list + data, axis=1)
    return tf.contrib.layers.fully_connected(stacked, output_size,
              activation_fn = tf.nn.tanh,
              biases_initializer = tf.random_normal_initializer)

  #Generate public info from private bits
  def obfuscate_private(self, bits, reuse, name='obfuscate'):
    with tf.variable_scope(name, reuse=reuse):
      return self.process([bits], KEY_SIZE, OBFUSCATION_DEPTH, OBFUSCATION_MEMORY)
  
  #Generate shared key from your private bits and the shared message
  def construct_key(self, bits_private, bits_received, reuse, name='construct_key'):
    with tf.variable_scope(name, reuse=reuse):
      return self.process([bits_private, bits_received], KEY_SIZE, CONSTRUCTION_DEPTH, CONSTRUCTION_MEMORY)

  #Eve tries to guess the shared key given the two public values
  def eve_attack_shared(self, bits_A, bits_B, reuse, name='shared_E'):
    with tf.variable_scope(name, reuse=reuse):
      return self.process([bits_A, bits_B], KEY_SIZE, EVE_DEPTH, EVE_MEMORY)

  #Eve tries to guess A's private bits given A's values
  def eve_attack_private(self, bits_shared, reuse, name='private_E'):
    with tf.variable_scope(name, reuse=reuse):
      return self.process([bits_shared], KEY_SIZE, EVE_DEPTH, EVE_MEMORY)

  def bit_accuracy(self, str1, str2):
    return (KEY_SIZE + tf.reduce_sum(str1 * str2, axis=1))/2 # dot product, then adjust to range from 0 to KEY_SIZE

  #Takes a batch of keys in [batch, bits] and produces a penalty loss
  def key_diversity_penalty(self, keys):
    #Just compare each key to the next one. They shouldn't be too similar.
    trim1 = keys[1:,:]
    trim2 = keys[:-1,:]
    xor = trim1 * trim2 #A bunch of pairwise XORs

    #We don't really care about the bits being different once they're pretty far apart --
    #the important thing is just having general entropy in the string. So if we keep pushing
    #them apart, we're incentivizing the tanh's to saturate really hard, which is going to make
    #training hard. Taking max(xor, -0.81) means that if we have [0.9, -0.9], we're considering
    #those bits to be "different enough" 
    xor = tf.maximum(xor, -0.81)

    #It should be marked diverse as long as _one_ of the bits is different. Sounds like softmax
    def softmax(data): #Not quite the same as tf.nn.softmax
      return tf.log(tf.reduce_sum(tf.exp(data), axis=1))

    # different bits are -1, similar bits are 1. So we need softmin = -softmax(-xor).
    # scale factor of -2 means we get e^(2*2) ~= 55 factor difference of weight between different/same bits
    # (if this we much smaller we would get it trying to make _all_ the bits different)
    max_diff = -softmax(-2 * xor) #Lower is more different
    baseline = -2 - math.log(KEY_SIZE) #What is the score corresponding to perfectly different?

    return (tf.reduce_mean(max_diff) - baseline) * KEY_SIZE #multiplying by KEY_SIZE scales to other error terms

  def get_weight_decay(self, group):
    """L2 weight decay loss."""
    costs = []
    for var in tf.contrib.framework.get_trainable_variables(scope=group):
      costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def __init__(self):
    secret_A = self.secret_A = get_rand_bits(BATCH, KEY_SIZE)
    secret_B = get_rand_bits(BATCH, KEY_SIZE)

    with tf.variable_scope('side_AB'):
      public_A = self.obfuscate_private(secret_A, reuse=False)
      public_B = self.obfuscate_private(secret_B, reuse=True)  #Variables initialized in pervious call

      shared_A = self.construct_key(secret_A, public_B, reuse=False)
      shared_B = self.construct_key(secret_B, public_A, reuse=True)
 
    with tf.variable_scope('side_E'):
      shared_E = self.eve_attack_shared(public_A, public_B, reuse=False) #Eve tries to recover the shared key from the comms
      secret_E = self.eve_attack_private(public_A, reuse=False) #Eve tries to guess the secret seed

    shared_agreed = (shared_A + shared_B)/2 # take the average of their two keys to get the 'agreed' key

    #Do the keys vary?
    #diversity_penalty says the generated keys should look random.
    self.diversity_penalty = self.key_diversity_penalty(shared_agreed)
    #public_diversity_penalty says that the public messages should look random as well.
    #Not actually sure if this helps, but it should make the network easier to train, since
    #to have a good scheme it is ultimately necessary to have KEY_SIZE bits of entropy in the
    #intermediate message too.
    self.public_diversity_penalty = (self.key_diversity_penalty(public_A) + self.key_diversity_penalty(public_B))/2
    
    #Do alice and bob get a diverse set of keys?
    self.agreed_count = tf.reduce_mean( self.bit_accuracy(shared_A, shared_B) )

    #Can eve guess the shared key?
    self.eve_count1 = tf.reduce_mean( self.bit_accuracy(shared_agreed, shared_E) )
    #Can eve guess the seed?
    self.eve_count2 = tf.reduce_mean( self.bit_accuracy(secret_A, secret_E) )

    self.eve_err_1 = KEY_SIZE - self.eve_count1
    self.eve_err_2 = KEY_SIZE - self.eve_count2
    self.eve_count = self.eve_count2
    self.eve_loss  = self.eve_err_1**2 + self.eve_err_2**2
    self.eve_loss += FLAGS.eve_decay * self.get_weight_decay('side_E')
    
    eve_guess_penalty_1 = tf.maximum(KEY_SIZE/2, self.eve_count1) ** 2 - (KEY_SIZE**2/4) #basically eve_count_1 ** 2
    eve_guess_penalty_2 = tf.maximum(KEY_SIZE/2, self.eve_count2) ** 2 - (KEY_SIZE**2/4) #basically eve_count_2 ** 2

    self.alice_bob_err = KEY_SIZE - self.agreed_count

    self.alice_bob_loss  =                      self.alice_bob_err ** 2
    self.alice_bob_loss += FLAGS.ab_diversity * (self.diversity_penalty**2 + self.public_diversity_penalty**2)
    self.alice_bob_loss += FLAGS.ab_eve1      * eve_guess_penalty_1
    self.alice_bob_loss += FLAGS.ab_eve2      * eve_guess_penalty_2
    self.alice_bob_loss += FLAGS.ab_decay     * self.get_weight_decay('side_AB')

    optimizer_AB = tf.train.MomentumOptimizer(learning_rate=FLAGS.ab_learn, momentum=0.9)
    self.train_AB = optimizer_AB.minimize(self.alice_bob_loss, var_list = tf.contrib.framework.get_variables('side_AB'))

    optimizer_E = tf.train.MomentumOptimizer(learning_rate=FLAGS.eve_learn, momentum=0.9)
    self.train_E = optimizer_E.minimize(self.eve_loss, var_list = tf.contrib.framework.get_variables('side_E'))

    self.shared_samples = shared_agreed

#end model definition

def train_and_evaluate():
  """Run the full training and evaluation loop."""
  model = SharedKeyCrypto()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    print('# Batch size: ', FLAGS.batch_size)
    print('# Iter AB_Error AB_Loss E_Error')

    AB_steps = 0
    E_steps = 0

    ITER = 3500
    start_time = time.time()
    for j in xrange(1,ITER):
      #AB_err: how many bits Alice/Bob are not matching correctly in their generated key
      #E_err: how many bis Eve is getting wrong (of the key, or the seed, depending on eve_count1 or eve_count2 abvoe)
      #keys are the keys generated in this batch
      #diversity_penalty is an estimate for how much "diversity" is lacking in the keys
      #alice_bob_loss is an overall score of A/B's role, trying to minimize their own error, maximize Eve's, increase diversity.
      AB_err, E_err_1, E_err_2, keys, diversity_P, AB_loss = sess.run(
      [model.alice_bob_err, model.eve_err_1, model.eve_err_2, model.shared_samples, model.diversity_penalty, model.alice_bob_loss])

      #Run one step of Alice/Bob, then two steps of Eve.
      #Once we're halfway through the iteration count, fix Alice/Bob and let try attacking.
      if j < ITER/2:
        sess.run(model.train_AB)
      sess.run(model.train_E)
      if j % 50 == 1 or j==ITER-1:
        print("%d\tEerr1=%.2f, Eerr2=%.2f, ABerr=%.2f. DivP=%.2f. Loss = %.2f (t=%.2f)" % (j, E_err_1, E_err_2, AB_err, diversity_P, AB_loss, time.time()-start_time))

      #Alternate training schedule where Alice/Bob trains if either
      # * They're getting more than one bit wrong or
      # * Eve is doing more than one bit better than random guessing
      # .. and otherwise Eve trains
      """if AB_err > 1 or KEY_SIZE/2 - 1 > E_err:
        sess.run(model.train_AB)
        AB_steps += 1
      else:
        sess.run(model.train_E)
        E_steps += 1
      print("%Eve's fraction of steps: %.2f. AB's fraction of steps: %.2f." % (E_steps /j, AB_steps / j)"""

    print("Some keys:")
    for i in range(5):
      pretty(keys[i])
    print("Some XORs between successive keys:")
    for i in range(5):
      pretty(keys[i] * keys[i+1])

def pretty(v):
  print("[",*list(map(lambda x:"%.1f,\t"%x, v)),"]")

def main(unused_argv):
  # Exit more quietly with Ctrl-C.
  signal.signal(signal.SIGINT, signal.SIG_DFL)
  train_and_evaluate()

if __name__ == '__main__':
  tf.app.run()

import sys
import numpy as np
from random import shuffle
import tensorflow as tf

"""
dataX = [
    '11010101111010001010',
    '11001111110010001111',
    '11101011111110110010',
    ...
]
dataX.shape == (None,)
"""
BATCH_SIZE = 100
SEQ_LENGTH = 12
NUM_LSTM_CELLS = 3
NUM_CLASSES = SEQ_LENGTH + 1
MAX_INT = 2 ** SEQ_LENGTH

# create all binary strings of length SEQ_LENGTH
dataX = ['{0:b}'.format(i).zfill(SEQ_LENGTH) for i in range(MAX_INT)]
shuffle(dataX)
# print "dataX", MAX_INT, len(dataX), dataX[0:10]

# Convert each string i into a list of binary digits
dataX = [map(int, i) for i in dataX]
# print "dataX", dataX[0:10]

"""
Convert training input into an array of shape (None, SEQ_LENGTH, 1):
dataX = [
    [[1], [0], [1], [1],...], # each row has length SEQ_LENGTH
    [[0], [0], [0], [1],...],
    [[1], [1], [1], [1],...],
    ...
]
dataX.shape == (None, SEQ_LENGTH, 1)
"""
ti = []
for i in dataX:
    temp_list = []
    for j in i:
        temp_list.append([j])
    ti.append(np.array(temp_list))
dataX = ti
# print "dataX", dataX[0:10]

"""
Create training targets: count number of ones in each training case. Each row is
a one-hot representation of the number (possible values 0 to
SEQ_LENGTH) of ones in the corresponding dataX row.

dataY = [
    [0,..., 0, 1, 0,..., 0],
    [1,..., 0, 0, 0,..., 0],
    [0,..., 0, 0, 0,..., 1],
    ...
]
dataY.shape == (None, NUM_CLASSES)
"""
dataY = []
for i in dataX:
    count = 0
    for j in i:
        if j[0] == 1:
            count += 1
    temp_list = ([0] * NUM_CLASSES)
    temp_list[count] = 1
    dataY.append(temp_list)
# print "dataY", dataY[0:10]

"""
Separating training and test data.
"""
NUM_EXAMPLES = int(0.9 * MAX_INT)
trainX = dataX[:NUM_EXAMPLES]
trainY = dataY[:NUM_EXAMPLES]
testX = dataX[NUM_EXAMPLES:]
testY = dataY[NUM_EXAMPLES:]
# print "MAX_INT", MAX_INT, "NUM_EXAMPLES", NUM_EXAMPLES
# print "num_train", len(trainX)
# print "num_test", len(testX)

"""
Placeholders for minibatch input and output data
"""
X = tf.placeholder(tf.float32, [None, SEQ_LENGTH, 1])
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

"""
BUILDING THE NETWORK

Output has shape (100, SEQ_LENGTH, NUM_CELL_UNITS). Each array output[i]
(SEQ_LENGTH, NUM_CELL_UNITS) contains the activations for a training case. Each
vector output[i][j] contains the activations after digit j has been read.
output[i][SEQ_LENGTH - 1] contains the activations after all digits in this
training case has been read. At this point, we have all the information required
to count the number of digits, so we take these final activations and map them
to class scores via a fully connected layer given by weights W and biases b.

output = [
    [
        [ 0.0016465  -0.00632819  0.00610369 -0.00471251  0.00714036  0.00548093
          0.0055373  -0.00520443  0.00535085  0.00700709  0.00587014  0.0062449
         -0.00259987 -0.00489188  0.00614024  0.00377309  0.00627311  0.00648128
         -0.00569265 -0.00162724 -0.00643665 -0.00259952 -0.00276687 -0.00159421]
        [ 0.00240126 -0.01185386  0.01099987 -0.00917767  0.01317046  0.01008037
          0.01109973 -0.00893197  0.00978673  0.01339379  0.01135415  0.01160987
         -0.00319728 -0.00846428  0.01126003  0.00683881  0.01196266  0.01327974
         -0.00975067 -0.00309993 -0.01243167 -0.00306832 -0.00396505 -0.00271353]
        [ 0.05784487  0.01532166 -0.00415068 -0.01310475  0.05880595  0.06590836
          0.02808037  0.00239125 -0.03848059  0.08118987 -0.02125091 -0.01840056
          0.0315199  -0.04297646  0.00695085 -0.03154649  0.06774446  0.04589796
         -0.05864895  0.03737433 -0.07135426 -0.02367695  0.02990603 -0.00122886]
        [ 0.04503402 -0.00136316  0.00051303 -0.02159743  0.05941822  0.05964475
          0.04088331 -0.00468607 -0.02227866  0.07906292  0.00577184  0.00633974
          0.02000848 -0.02801351  0.01559633 -0.01900019  0.04777982  0.04801929
         -0.05231006  0.02545839 -0.06967451 -0.0250156   0.02092477 -0.00217354]
        [ 0.03785722 -0.01236351  0.00626796 -0.0290583   0.0556293   0.05577802
          0.04905227 -0.00878377 -0.0115157   0.07637603  0.02425439  0.0217574
          0.01340329 -0.01730466  0.02216251 -0.01189383  0.03838311  0.0506798
         -0.04330656  0.01227054 -0.0683573  -0.02255086  0.01331724 -0.00208541]
        [ 0.08693553  0.01143604 -0.00750642 -0.03489176  0.09103136  0.10436653
          0.06896032  0.00318358 -0.05448574  0.13834293  0.00233034 -0.00513268
          0.04378038 -0.04367679  0.01881823 -0.04778607  0.08649891  0.08067303
         -0.07972414  0.04380088 -0.11668894 -0.03902884  0.04095824  0.00072619]
        [ 0.06835961 -0.00627753 -0.0021984  -0.0430135   0.08488343  0.08924159
          0.07732179 -0.00333198 -0.03392861  0.12726456  0.03195626  0.02038329
          0.03182886 -0.02549104  0.02799471 -0.02983429  0.0578646   0.07886841
         -0.0702939   0.02581234 -0.10932986 -0.03623104  0.03104804  0.00142149]
        [ 0.05828394 -0.01755368  0.0047556  -0.05039673  0.07416295  0.07940711
          0.08196895 -0.00709579 -0.02081483  0.11705207  0.05234597  0.03428879
          0.02409048 -0.01345155  0.03460353 -0.01993554  0.04342943  0.07783701
         -0.05714302  0.00795002 -0.10349434 -0.03033715  0.02126135  0.0025966 ]
        [ 0.05012789 -0.02507182  0.01080788 -0.05533992  0.06392521  0.06992313
          0.08320192 -0.0091367  -0.01092021  0.10790394  0.0661767   0.04084911
          0.02059685 -0.00822499  0.03964521 -0.01112671  0.03478388  0.07718994
         -0.04780928 -0.00549135 -0.09846148 -0.02304958  0.01494825  0.00401088]
        [ 0.04377255 -0.03010502  0.01595715 -0.05849586  0.05512481  0.06125101
          0.08282741 -0.01041711 -0.00352233  0.09990177  0.07497081  0.04292843
          0.01921426 -0.00705276  0.04356502 -0.003192    0.02999888  0.0767495
         -0.04106682 -0.01536205 -0.0945136  -0.01552995  0.01043605  0.00558612]
        [ 0.03899902 -0.03357863  0.02026168 -0.0603694   0.04805423  0.05363467
          0.08175934 -0.01138794  0.00198381  0.09302361  0.08010822  0.04247135
          0.01878729 -0.00822247  0.04657086  0.00395387  0.02779079  0.07641925
         -0.03602442 -0.02240469 -0.09162445 -0.00840751  0.00681478  0.00726885]
        [ 0.03554999 -0.0361334   0.02379958 -0.06134978  0.0426457   0.04716176
          0.08049704 -0.0122334   0.00609409  0.08721898  0.0827103   0.04076022
          0.01872697 -0.01064561  0.04879219  0.01035207  0.02729457  0.07616398
         -0.03209597 -0.02724671 -0.08964177 -0.0019863   0.00362077  0.00900897]
        [ 0.03316326 -0.03818462  0.02665696 -0.06171732  0.03867675  0.04180697
          0.07929311 -0.01300863  0.00920017  0.08240612  0.08364873  0.03860225
          0.01875561 -0.01363288  0.05033172  0.01604049  0.02792072  0.07597382
         -0.02891051 -0.03040569 -0.08838133  0.0036166   0.00064357  0.01075725]
        [ 0.08605256 -0.00889792  0.00817091 -0.06322533  0.07654381  0.09033494
          0.09588683  0.00063931 -0.04009559  0.14324142  0.0537244   0.00085435
          0.05107512 -0.04807122  0.04262343 -0.0199828   0.08169425  0.10309532
         -0.06975457  0.01188309 -0.13608685 -0.01317484  0.03166328  0.01513525]
    ],
    ...
]
"""
NUM_CELL_UNITS = 25  # poij. Adjust hyperparam
Cell = tf.contrib.rnn.BasicLSTMCell(NUM_CELL_UNITS, state_is_tuple=True)
Cells = tf.contrib.rnn.MultiRNNCell([Cell] * NUM_LSTM_CELLS)

"""
State is a tuple of (NUM_LSTM_CELLS) cell states, one state for each cell. E.g.
State = ((c, h), (c, h), (c, h)), where c is the cell state and h is the cell's
hidden state. Also State[-1].h == Output[:, -1, :] == Last.
"""
Output, State = tf.nn.dynamic_rnn(Cells, X, dtype=tf.float32)
Output = tf.reshape(Output, [tf.shape(Output)[0], -1])

"""
Fully connected layer mapping RNN output to classes
"""
W = tf.Variable(tf.truncated_normal(
    [SEQ_LENGTH * NUM_CELL_UNITS, NUM_CLASSES]))
b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
Scores = tf.matmul(Output, W) + b

"""
Loss and optimization
"""
Pred = tf.nn.softmax(Scores)
Loss = - tf.reduce_mean(tf.reduce_sum(Y * tf.log(Pred)))  # cross entropy
Minimize = tf.train.AdamOptimizer().minimize(Loss)
Corrects = tf.equal(tf.argmax(Y, axis=1), tf.argmax(Pred, axis=1))
Accuracy = tf.reduce_mean(tf.cast(Corrects, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

"""
Training
"""
num_batches = int(len(trainX) / BATCH_SIZE)
# epochs = 100
epochs = 50
# epochs = 10
for i in range(epochs):
    print "Epoch:", i
    ptr = 0
    for j in range(num_batches):
        batchX = trainX[ptr:ptr + BATCH_SIZE]
        batchY = trainY[ptr:ptr + BATCH_SIZE]
        minimize, output, state = sess.run(
            [Minimize, Output, State], {X: batchX, Y: batchY})

        ptr += BATCH_SIZE

        if j % 10 == 0:
            accuracy = sess.run(Accuracy, {X: batchX, Y: batchY})
            print('Batch: {:2d}, accuracy: {:3.1f} %'.format(
                j, 100 * accuracy))

"""
Testing
"""
accuracy = sess.run(Accuracy, {X: testX, Y: testY})
print('Accuracy on test set: {:3.1f} %'.format(100 * accuracy))

# Don't need to pass Y in here because Pred doesn't need it:
pred = sess.run(Pred, {X: testX})
print 'Test case:', testX[:1]
print 'Prediction:', pred[:1]

sess.close()

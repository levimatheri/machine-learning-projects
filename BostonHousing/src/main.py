# Author: Levi Muriuki

from boston import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scores import *


def main():
    # obtain dataframe
    df = get_data('../data.csv')

    # remove skewness
    df = remove_skew(df)

    # scale data
    df = normalize(df)

    x = pd.DataFrame(data=df.drop(columns=['medv']))
    y = df['medv']

    # split data to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    y_train = y_train.values.reshape(-1, 1)
    y_test = y_test.values.reshape(-1, 1)

    num_features = x_train.shape[1]

    inputs = tf.placeholder(tf.float32, shape=(None, num_features), name='X_in')
    outputs = tf.placeholder(tf.float32, shape=(None, 1), name='Y_out')

    # create variables
    batch_size = 100
    num_batches = int(x_train.shape[0] / batch_size)

    y_hat = model(inputs, num_features)
    learning_rate = 0.001

    cost_op = tf.reduce_mean(tf.pow(y_hat - outputs, 2))
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost_op)

    total_epochs = 20000
    # train model
    sess = tf.Session()
    with sess.as_default():
        # initialize vars
        sess.run(tf.global_variables_initializer())

        costs = []
        epochs = []
        epoch = 0
        # train until epochs total
        while True:
            cost = 0.0
            for n in range(num_batches):
                x_batch = x_train[n * batch_size: (n + 1) * batch_size]
                y_batch = y_train[n * batch_size: (n + 1) * batch_size]

                sess.run(train, feed_dict={inputs: x_batch, outputs: y_batch})
                c = sess.run(cost_op, feed_dict={inputs: x_batch, outputs: y_batch})
                cost += c
            cost /= num_batches
            costs.append(cost)
            epochs.append(epoch)
            epoch += 1

            if epoch % 1000 == 0:
                print("Cost after %d epochs: %1.8f" % (epoch, cost))
            if epoch >= total_epochs:
                break

        # plot training cost
        plot_train_cost(epochs, costs)

        # make some predictions
        y_pred = sess.run(y_hat, feed_dict={inputs: x_test, outputs: y_test})

        print("\nPrediction\nreal\tpredicted")
        for (y, y_) in list(zip(y_test, y_pred))[0:10]:
            print("%1.1f\t%1.1f" % (y, y_))

    # print scores
    print_metrics(y_test, y_pred)


def plot_train_cost(epochs, costs):
    plt.figure()
    plt.title('Training cost plot')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(epochs, costs, color='r', label='Training cost')


if __name__ == '__main__':
    main()

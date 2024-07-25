import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace


class MyTrigram(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size=100, embed_size=64):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Use Embedding layer for the embedding matrix
        self.embedding_layer = tf.keras.layers.Embedding(
            vocab_size,
            embed_size,
            embeddings_initializer=tf.random_normal_initializer(stddev=0.01),
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(50, activation="relu")
        self.dense2 = tf.keras.layers.Dense(embed_size, activation="softmax")

    def call(self, inputs):
        """
        :param inputs: word ids of shape (batch_size, 2)
        :return: logits: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        embedding_vectors = self.embedding_layer(inputs)
        nicely_shaped_embeddings = self.flatten(embedding_vectors)
        x = self.dense1(nicely_shaped_embeddings)
        x = self.dense2(x)
        return x

    def generate_sentence(self, word1, word2, length, vocab):
        """
        Given initial 2 words, print out predicted sentence of targeted length.
        (NOTE: you shouldn't need to make any changes to this function).

        :param word1: string, first word
        :param word2: string, second word
        :param length: int, desired sentence length
        :param vocab: dictionary, word to id mapping

        """
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int32)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            output_string[:, end] = np.argmax(self(output_string[:, start:end]), axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


def perplex(labels, preds):
    calculate_categorical = tf.keras.losses.SparseCategoricalCrossentropy()(
        labels, preds
    )
    # tf.print("labels shape", labels.shape)
    # tf.print("preds shape", preds.shape)
    # tf.print("calculate categorical: ", calculate_categorical)
    divide_by_batch_size = tf.reduce_mean(calculate_categorical)
    # tf.print("divide by batch size: ", divide_by_batch_size)
    tf.print("exp: ", tf.exp(divide_by_batch_size))
    return tf.exp(divide_by_batch_size)


#########################################################################################


def get_text_model(vocab):
    """
    Tell our autograder how to train and test your model!
    """

    ## TODO: Set up your implementation of the RNN

    ## Optional: Feel free to change or add more arguments!
    model = MyTrigram(len(vocab))

    ## TODO: Define your own loss and metric for your optimizer
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric = perplex

    # Sanity check for the perplexity calculation
    random_pred = tf.Variable(
        np.array(
            [
                [0.1, 0.3, 0.5, 0.1],
                [0.4, 0.3, 0.1, 0.2],
                [0.1, 0.7, 0.1, 0.1],
                [0.3, 0.3, 0.2, 0.2],
            ]
        )
    )
    random_true = tf.Variable(np.array([2, 0, 1, 3]))
    np.testing.assert_almost_equal(
        np.mean(acc_metric(random_true, random_pred)), 2.4446151121745054, decimal=4
    )
    print("returned true")

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss_metric,
        metrics=[acc_metric],
    )

    return SimpleNamespace(
        model=model,
        epochs=1,
        batch_size=128,
    )


#########################################################################################


def main():

    ## TODO: Pre-process and vectorize the data
    ##   HINT: You might be able to find this somewhere...
    # data_path = 'C:/dl/homework-4p-language-models-manolodalbo/data'
    data_path = "../data"
    train_tokens, test_tokens, vocab = get_data(
        f"{data_path}/train.txt", f"{data_path}/test.txt"
    )
    train_array = np.array(train_tokens)
    test_array = np.array(test_tokens)
    X0, Y0 = np.vstack([train_array[0:-2], train_array[1:-1]]).T, train_array[2:]
    X1, Y1 = np.vstack([test_array[0:-2], test_array[1:-1]]).T, test_array[2:]

    # Sanity Check!
    assert X0.shape[1] == 2
    assert X1.shape[1] == 2
    assert X0.shape[0] == Y0.shape[0]
    assert X1.shape[0] == Y1.shape[0]
    perplex_history = []
    loss_history = []

    class PerplexHistory(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            perplex = logs.get("perplex")
            perplex_history.append(perplex)
            loss_history.append(logs.get("loss"))
            print(f"Batch {batch}, Perplex: {perplex}")
            print(f"perplex mean: {np.mean(np.array(perplex_history))}")

    # TODO: Implement get_text_model to return the model that you want to use.
    args = get_text_model(vocab)
    args.model.fit(
        X0,
        Y0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X1, Y1),
        callbacks=[PerplexHistory()],
    )
    print(args.model.metrics)

    ## Feel free to mess around with the word list to see the model try to generate sentences
    words = "speak to this brown deep learning student".split()
    for word1, word2 in zip(words[:-1], words[1:]):
        if word1 not in vocab:
            print(f"{word1} not in vocabulary")
        if word2 not in vocab:
            print(f"{word2} not in vocabulary")
        else:
            args.model.generate_sentence(word1, word2, 20, vocab)


if __name__ == "__main__":
    main()

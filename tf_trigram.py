import tensorflow as tf
import numpy as np
from preprocess import get_data
from types import SimpleNamespace
import SpinTorch.spintorch as spintorch
import os


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
        self.dense2 = tf.keras.layers.Dense(vocab_size, activation="softmax")

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
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        output_string = np.zeros((1, length), dtype=np.int32)
        output_string[:, :2] = vocab[word1], vocab[word2]

        for end in range(2, length):
            start = end - 2
            pred = self(np.array(output_string[:, start:end], dtype=np.int32))
            output_string[:, end] = np.argmax(pred, axis=1)
        text = [reverse_vocab[i] for i in list(output_string[0])]

        print(" ".join(text))


def perplexity(labels, preds):
    """
    Compute the perplexity of predictions.
    :param labels: ground truth labels
    :param preds: predictions
    :return: perplexity value
    """
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    loss = cross_entropy(labels, preds)
    return tf.exp(tf.reduce_mean(loss))


def get_text_model(vocab):
    """
    Create and compile the text model.
    :param vocab: vocabulary size
    :return: model and training configuration
    """
    model = MyTrigram(len(vocab))

    # Define loss and metric
    loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    acc_metric = perplexity

    # Sanity check for the perplexity calculation
    random_pred = tf.Variable(
        np.array(
            [
                [0.1, 0.3, 0.5, 0.1],
                [0.4, 0.3, 0.1, 0.2],
                [0.1, 0.7, 0.1, 0.1],
                [0.3, 0.3, 0.2, 0.2],
            ]
        ),
        dtype=tf.float32,
    )
    random_true = tf.Variable(np.array([2, 0, 1, 3]), dtype=tf.int32)
    print(acc_metric(random_true, random_pred).numpy())
    np.testing.assert_almost_equal(
        np.mean(acc_metric(random_true, random_pred).numpy()),
        2.4446151121745054,
        decimal=4,
    )
    print("returned true")

    return SimpleNamespace(
        model=model,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss_metric=loss_metric,
        acc_metric=acc_metric,
        epochs=3,
        batch_size=128,
    )


def main():
    data_path = "../data"
    train_tokens, test_tokens, vocab = get_data(
        f"{data_path}/train.txt", f"{data_path}/test.txt"
    )
    train_array = np.array(train_tokens)
    test_array = np.array(test_tokens)
    X0, Y0 = np.vstack([train_array[0:-2], train_array[1:-1]]).T, train_array[2:]
    X1, Y1 = np.vstack([test_array[0:-2], test_array[1:-1]]).T, test_array[2:]
    X0 = tf.convert_to_tensor(X0, dtype=tf.int32)
    X1 = tf.convert_to_tensor(X1, dtype=tf.int32)
    Y0 = tf.convert_to_tensor(Y0, dtype=tf.int32)
    Y1 = tf.convert_to_tensor(Y1, dtype=tf.int32)

    # Sanity Checks
    assert X0.shape[1] == 2
    assert X1.shape[1] == 2
    assert X0.shape[0] == Y0.shape[0]
    assert X1.shape[0] == Y1.shape[0]

    args = get_text_model(vocab)

    basedir = "focus_Ms/"
    plotdir = "plots/" + basedir
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    loss_history = []
    to_plot = []
    for epoch in range(args.epochs):
        perplexities = []
        for b in range(0, X0.shape[0], args.batch_size):
            b0 = b
            b1 = min(b + args.batch_size, X0.shape[0])
            with tf.GradientTape() as tape:
                pred = args.model(X0[b0:b1])
                loss = args.loss_metric(Y0[b0:b1], pred)
            gradients = tape.gradient(loss, args.model.trainable_variables)
            args.optimizer.apply_gradients(
                zip(gradients, args.model.trainable_variables)
            )

            perplex = args.acc_metric(Y0[b0:b1], pred)
            perplexities.append(perplex.numpy())
            loss_history.append(loss.numpy())
        print(f"Epoch {epoch + 1} perplexity: {np.mean(perplexities)}")

    spintorch.plot.plot_loss(loss_history, plotdir, "tf_trigram_manual")

    # Generate sentences
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

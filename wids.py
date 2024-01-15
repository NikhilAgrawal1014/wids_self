
import numpy as np
import os
import random
import re
import shutil
import time
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector


class LstmRNN(object):
    def __init__(self, sess, stock_count,
                 lstm_size=128,
                 num_layers=1,
                 num_steps=30,
                 input_size=1,
                 embed_size=None,
                 logs_dir="logs",
                 plots_dir="images"):
        """
        Construct a RNN model using LSTM cell.

        Args:
            sess:
            stock_count (int): num. of stocks we are going to train with.
            lstm_size (int)
            num_layers (int): num. of LSTM cell layers.
            num_steps (int)
            input_size (int)
            keep_prob (int): (1.0 - dropout rate.) for a LSTM cell.
            embed_size (int): length of embedding vector, only used when stock_count > 1.
            checkpoint_dir (str)
        """
        self.sess = sess
        self.stock_count = stock_count

        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.input_size = input_size

        self.use_embed = (embed_size is not None) and (embed_size > 0)
        self.embed_size = embed_size or -1

        self.logs_dir = logs_dir
        self.plots_dir = plots_dir

        self.build_graph()

    def build_graph(self):
        """
        The model asks for five things to be trained:
        - learning_rate
        - keep_prob: 1 - dropout rate
        - symbols: a list of stock symbols associated with each sample
        - input: training data X
        - targets: training label y
        """
        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        # Stock symbols are mapped to integers.
        self.symbols = tf.placeholder(tf.int32, [None, 1], name='stock_labels')

        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.input_size], name="targets")

     """
     TensorFlow placeholders are used to feed data into the computational graph during training or inference.
     several placeholder ara as: learning_rate, keep_prob, symbols, inputs, targets.
     None values in the shape argument of placeholders indicate that the corresponding dimensions can vary during runtime, 
     allowing the model to handle variable batch sizes.
     Stock symbols are expected to be passed as integers through the symbols placeholder.
     targets placeholder is used for the target values the model is trying to predict. """

        def _create_one_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.lstm_size, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple=True
        ) if self.num_layers > 1 else _create_one_cell()

        if self.embed_size > 0 and self.stock_count > 1:
            self.embed_matrix = tf.Variable(
                tf.random_uniform([self.stock_count, self.embed_size], -1.0, 1.0),
                name="embed_matrix"
            )

            """
            _create_one_cell Method: This method is defined to create an LSTM cell. It uses TensorFlow's LSTMCell 
            and wraps it with a DropoutWrapper to apply dropout regularization during training.
            MultiRNNCell for Multiple Layers: The code then uses the _create_one_cell method to create either a single 
            LSTM cell or a multi-layered LSTM cell (MultiRNNCell).
            Embedding Matrix: If embedding is specified and there is more than one stock an embedding matrix is created"""
            
            # stock_label_embeds.shape = (batch_size, embedding_size)
            stacked_symbols = tf.tile(self.symbols, [1, self.num_steps], name='stacked_stock_labels')
            stacked_embeds = tf.nn.embedding_lookup(self.embed_matrix, stacked_symbols)

            # After concat, inputs.shape = (batch_size, num_steps, input_size + embed_size)
            self.inputs_with_embed = tf.concat([self.inputs, stacked_embeds], axis=2, name="inputs_with_embed")
            self.embed_matrix_summ = tf.summary.histogram("embed_matrix", self.embed_matrix)

        else:
            self.inputs_with_embed = tf.identity(self.inputs)
            self.embed_matrix_summ = None

        print "inputs.shape:", self.inputs.shape
        print "inputs_with_embed.shape:", self.inputs_with_embed.shape
    
      """
      Summary Operation for Embedding Matrix:
      It creates a summary operation (tf.summary.histogram) for the embedding matrix, which can be useful for 
       visualizing and tracking the distribution of values in the embedding matrix during training.
      If embedding is not used, self.embed_matrix_summ is set to None.
Print Statements: The code includes two print statements for informational purposes, displaying the shapes of self.inputs and self.inputs_with_embed"""

        # Run dynamic RNN
        val, state_ = tf.nn.dynamic_rnn(cell, self.inputs_with_embed, dtype=tf.float32, scope="dynamic_rnn")

        # Before transpose, val.get_shape() = (batch_size, num_steps, lstm_size)
        # After transpose, val.get_shape() = (num_steps, batch_size, lstm_size)
        val = tf.transpose(val, [1, 0, 2])

        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.input_size]), name="w")
        bias = tf.Variable(tf.constant(0.1, shape=[self.input_size]), name="b")
        self.pred = tf.matmul(last, ws) + bias

        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("w", ws)
        self.b_sum = tf.summary.histogram("b", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        """
        It utilizes the dynamic RNN  to process the input sequences using the configured LSTM cell (cell).
        The output tensor val is transposed to have dimensions in the order of (num_steps, batch_size, lstm_size).
        The last time step output (last) is extracted from the transposed tensor.
        Weights (ws) and bias (bias) are defined as trainable variables.
        The final prediction (self.pred) is calculated by multiplying the last time step output with weights and adding the bias.
        Summary histograms are created for the LSTM state, weights, bias, and predictions for visualization purposes during training."""

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")

        # Separated from train loss.
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    """
    Loss Calculation:
    Mean squared error is used as the loss function to measure the difference between the predicted values and the actual targets.
    The test loss is also computed using the same mean squared error metric.
    Optimization: RMSProp optimizer is employed to minimize the training loss.
    Summaries: TensorBoard summaries are created for the training loss, test loss, and learning rate.
    These summaries are useful for visualizing and monitoring the training process.
    Trainable Variables and Saver:Trainable variables are collected, and a TensorFlow Saver is created to save and 
    restore the model during training and evaluation."""

    def train(self, dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        assert len(dataset_list) > 0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./logs", self.model_name))
        self.writer.add_graph(self.sess.graph)

        if self.use_embed:
            # Set up embedding visualization
            # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
            projector_config = projector.ProjectorConfig()

            # You can add multiple embeddings. Here we add only one.
            added_embed = projector_config.embeddings.add()
            added_embed.tensor_name = self.embed_matrix.name
            # Link this tensor to its metadata file (e.g. labels).
            shutil.copyfile(os.path.join(self.logs_dir, "metadata.tsv"),
                            os.path.join(self.model_logs_dir, "metadata.tsv"))
            added_embed.metadata_path = "metadata.tsv"

            # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
            # read this file during startup.
            projector.visualize_embeddings(self.writer, projector_config)

        tf.global_variables_initializer().run()
"""
     Dataset List Validation: It checks if the provided list of datasets is not empty.
     Summary Merging: self.merged_sum is created by merging all summaries. T
     Summary Writer: A tf.summary.FileWriter is initialized to write the summaries to the specified logs directory.
     The graph of the TensorFlow session (self.sess.graph) is added to the writer for visualization in TensorBoard.
     Embedding Visualization (if applicable): If the model uses embedding, it sets up the configuration for embedding visualization.
     Global Variable Initialization: Global variables are initialized to prepare the model for training or evaluation.
"""
        # Merged test data of different stocks.
        merged_test_X = []
        merged_test_y = []
        merged_test_labels = []

        for label_, d_ in enumerate(dataset_list):
            merged_test_X += list(d_.test_X)
            merged_test_y += list(d_.test_y)
            merged_test_labels += [[label_]] * len(d_.test_X)

        merged_test_X = np.array(merged_test_X)
        merged_test_y = np.array(merged_test_y)
        merged_test_labels = np.array(merged_test_labels)

        print "len(merged_test_X) =", len(merged_test_X)
        print "len(merged_test_y) =", len(merged_test_y)
        print "len(merged_test_labels) =", len(merged_test_labels)

    # Data Merging: Three lists are initialized to store the merged test data.
    # The code then iterates over the provided dataset list and concatenates the test data from each dataset to the corresponding merged lists.
    # merged_test_labels is created as a list of labels for each sample, where each label corresponds to the index of the dataset in the dataset_list.
    # Finally, the lists are converted to NumPy arrays 
    # Print Lengths: It prints the lengths of the merged test data arrays for verification purposes.


        test_data_feed = {
            self.learning_rate: 0.0,
            self.keep_prob: 1.0,
            self.inputs: merged_test_X,
            self.targets: merged_test_y,
            self.symbols: merged_test_labels,
        }

        global_step = 0

        num_batches = sum(len(d_.train_X) for d_ in dataset_list) // config.batch_size
        random.seed(time.time())
 
    # Test Data Feed:test_data_feed is a dictionary containing placeholders and their corresponding values for the evaluation.
    # global_step is set to 0, keeping track of the total optimization steps across training epochs.
    # Number of Batches: num_batches is calculated as the total number of batches in the training datasets. 
    # A random seed is initialized based on the current time, ensuring reproducibility when using random functions in the code.
  
        # Select samples for plotting.
        sample_labels = range(min(config.sample_size, len(dataset_list)))
        sample_indices = {}
        for l in sample_labels:
            sym = dataset_list[l].stock_sym
            target_indices = np.array([
                i for i, sym_label in enumerate(merged_test_labels)
                if sym_label[0] == l])
            sample_indices[sym] = target_indices
        print sample_indices

        print "Start training for stocks:", [d.stock_sym for d in dataset_list]


    
     # Sample Labels:
     # sample_labels is a list containing indices of stocks from dataset_list selected for plotting. 
     # sample_indices is a dictionary used to store indices of samples for each selected stock. 
     # For each selected label (l) in sample_labels, it retrieves the stock symbol (sym) and 
     # finds the indices of samples in the merged test data where the stock label matches the current label.
     # Print Statements: The dictionary of sample indices (sample_indices) is printed for reference.
     

        for epoch in xrange(config.max_epoch):
            epoch_step = 0
            learning_rate = config.init_learning_rate * (
                config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for label_, d_ in enumerate(dataset_list):
                for batch_X, batch_y in d_.generate_one_epoch(config.batch_size):
                    global_step += 1
                    epoch_step += 1
                    batch_labels = np.array([[label_]] * len(batch_X))
                    train_data_feed = {
                        self.learning_rate: learning_rate,
                        self.keep_prob: config.keep_prob,
                        self.inputs: batch_X,
                        self.targets: batch_y,
                        self.symbols: batch_labels,
                    }
                    train_loss, _, train_merged_sum = self.sess.run(
                        [self.loss, self.optim, self.merged_sum], train_data_feed)
                    self.writer.add_summary(train_merged_sum, global_step=global_step)

  
    # Epoch Loop: The outer loop iterates over epochs, ranging from 0 to config.max_epoch.
    # Learning Rate Decay: The learning rate (learning_rate) is decayed based on the configured learning rate decay strategy.
    # Inner Loop - Dataset Iteration:For each stock dataset (d_) in dataset_list, the inner loop iterates over batches of training data 
    # (batch_X, batch_y) generated by the generate_one_epoch function of the dataset.
    # Training Data Feed:
    # A dictionary (train_data_feed) is created to feed data into the TensorFlow placeholders:
    # self.learning_rate is set to the current learning rate.
    # self.keep_prob is set to the configured dropout rate.
    # self.inputs receives the input sequences (batch_X).
    # self.targets receives the corresponding target values (batch_y).
    # self.symbols receives labels for the current stock dataset (batch_labels).
    # Training Operation: The model is trained for the current batch, and the training loss, optimization operation, 
    # and merged summaries are fetched from the TensorFlow session using self.sess.run.
    # Summary Logging: The training merged summary is added to the TensorBoard writer, and the global step is incremented.
   

                    if np.mod(global_step, len(dataset_list) * 200 / config.input_size) == 1:
                        test_loss, test_pred = self.sess.run([self.loss_test, self.pred], test_data_feed)

                        print "Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                            global_step, epoch, learning_rate, train_loss, test_loss)

                        # Plot samples
                        for sample_sym, indices in sample_indices.iteritems():
                            image_path = os.path.join(self.model_plots_dir, "{}_epoch{:02d}_step{:04d}.png".format(
                                sample_sym, epoch, epoch_step))
                            sample_preds = test_pred[indices]
                            sample_truth = merged_test_y[indices]
                            self.plot_samples(sample_preds, sample_truth, image_path, stock_sym=sample_sym)

                        self.save(global_step)

        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model
        self.save(global_step)
        return final_pred

  
  #Conditional Evaluation: The code checks if the current global step is such that it's time to perform evaluation based on the configured interval 
  #(len(dataset_list) * 200 / config.input_size). If true, the following steps are executed.
  #Test Data Evaluation: The test loss (test_loss) and predictions (test_pred) are obtained by running the TensorFlow session for the test data feed.
  #Print Progress: The script prints training progress information, including the current step, epoch, learning rate, training loss (train_loss), and test loss.
  #Plotting Samples: For each stock symbol in the sample indices, sample predictions and truths are extracted from the test predictions and actual test data, respectively. 
  #These samples are then plotted and saved as images in the model's plot directory using the plot_samples function.
  #Model Saving: The model is saved at this step using the save function, and the current global step is passed for checkpointing.
  #Final Evaluation: After completing the evaluation loop, a final evaluation is performed on the entire test dataset, and the final predictions and loss are obtained.
  #Save Final Model: The final model, trained on both training and test data, is saved, and the global step at which it was saved is returned.
  #Return Final Predictions: The final predictions on the test data are returned.


    @property
    def model_name(self):
        name = "stock_rnn_lstm%d_step%d_input%d" % (
            self.lstm_size, self.num_steps, self.input_size)

        if self.embed_size > 0:
            name += "_embed%d" % self.embed_size

        return name

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir

   
    # model_name Property: This property generates a unique name for the model based on its configuration parameters 
    # such as LSTM size (lstm_size), number of steps (num_steps), and input size (input_size).
    # If the model uses embedding (embed_size is greater than 0), it appends "_embed" followed by the embedding size to the model name.
    # model_logs_dir Property: This property returns the directory path where the model's logs will be stored.
    # It is created if it does not exist.
    # model_plots_dir Property: This property returns the directory path where the model's plots will be stored. 
    # It is created if it does not exist.
 

    def save(self, step):
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        
        
        #save Method: Takes a step parameter indicating the global step of the training.
        #  Constructs the model name by appending ".model" to the generated model name.
        #  Uses TensorFlow's Saver to save the model's parameters and states in the specified 
        #  directory (model_logs_dir) with the constructed model name and global step.
        #  load Method: Attempts to read the latest checkpoint from the model's log directory (model_logs_dir).
        # If a valid checkpoint is found, it restores the model's parameters and states from that checkpoint using TensorFlow's Saver.
       # Extracts the global step from the checkpoint name and returns a tuple indicating the success status and the counter at which the model was saved.
    # If no checkpoint is found, it prints a message indicating the failure and returns a tuple with a success status of False and a counter of 0.

    def plot_samples(self, preds, targets, figname, stock_sym=None, multiplier=5):
        def _flatten(seq):
            return np.array([x for y in seq for x in y])

        truths = _flatten(targets)[-200:]
        preds = (_flatten(preds) * multiplier)[-200:]
        days = range(len(truths))[-200:]

        plt.figure(figsize=(12, 6))
        plt.plot(days, truths, label='truth')
        plt.plot(days, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("day")
        plt.ylabel("normalized price")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if stock_sym:
            plt.title(stock_sym + " | Last %d days in test" % len(truths))

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
        plt.close()

        """
        This part of the code defines a method plot_samples within the LstmRNN class,
          which is responsible for creating a plot comparing predicted and true values.
           Input Parameters:
           preds: Predicted values., targets: True values. etc
           Functionality:
          _flatten function is defined to flatten a sequence of sequences (nested lists) into a 1D NumPy array.
           Extracts the last 200 data points from both the true and predicted values for plotting.
           Creates a line plot with two lines: one for the true values (truth) and one for the predicted values (pred), using Matplotlib.
           Includes labels, legend, and grid in the plot for better readability.
           If stock_sym is provided, the plot is titled with the stock symbol and the number of days in the test set.
           Saves the plot as a PNG file with the specified file name (figname).
           Closes the Matplotlib figure after saving."""
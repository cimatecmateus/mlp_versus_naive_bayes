import logging
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from os import sys

class MLP:
  def __init__(self):
    pass
  
  def get_data(self, data_path):
    self.data = pd.read_csv(data_path + 'cmc.data', header=None)
    
    map_dic = {'No-use': 1, 'Long-term': 2, 'Short-term': 3} 
    
    self._labels_list = []
    for key in map_dic:
      self._labels_list.append(key)
    self._labels_len = len(self._labels_list)

  def prepare_data(self, normalize_data=False):
    # Dataframe hold ID, 9 attributes and 1 class attribute (label)
    self._number_of_attributes = self.data.shape[1] - 1
    self._label_column = self._number_of_attributes

    dataset_x = self.data.iloc[:,list(range(0, self._number_of_attributes))]
    dataset_y = self.data.iloc[:,[self._number_of_attributes]]

    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(dataset_x.values, dataset_y.values, test_size=0.3)
    print(self.x_train.shape)
    print(self.y_train.shape)

    if normalize_data:
        max_value = np.max([self.x_train.max(), self.x_test.max()])
        self.x_train = self.x_train / max_value 
        self.x_test = self.x_test / max_value

    self.train_label = np.array([[(1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0)] for x in self.y_train])
    self.test_label = np.array([[(1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0)] for x in self.y_test])
  
  def create_model(self, hiden_layer_neurons, activation_functions, dropout_parameters, _loss, _metrics, _optimizer, lr):
    input_size = self._number_of_attributes
    self.neural_network_model = tf.keras.models.Sequential()
    self.neural_network_model.add(tf.keras.layers.Dense(input_size,
                                  input_shape=(input_size,),
                                  activation=activation_functions[0],
                                  kernel_initializer='he_uniform'))
    if dropout_parameters[0] == True:
      self.neural_network_model.add(tf.keras.layers.Dropout(dropout_parameters[1]))
    self.neural_network_model.add(tf.keras.layers.Dense(hiden_layer_neurons, activation=activation_functions[1]))
    if dropout_parameters[2] == True:
      self.neural_network_model.add(tf.keras.layers.Dropout(dropout_parameters[3]))
    self.neural_network_model.add(tf.keras.layers.Dense(1, activation=activation_functions[2]))

    self.neural_network_model.compile(loss=_loss,
                                      optimizer=self._get_optimizer_from_name(_optimizer, lr),
                                      metrics=_metrics)

  def train(self, _batch_size, _epochs):
    self.history = self.neural_network_model.fit(self.x_train, self.train_label,
                                                 batch_size=_batch_size,
                                                 epochs=_epochs,
                                                 verbose=1,
                                                 validation_data=(self.x_test, self.test_label))
    self.loss_value, self.accuracy_value = self.neural_network_model.evaluate(self.x_test, self.test_label)
    print("Loss value=", self.loss_value, "Accuracy value =", self.accuracy_value)
  
  def show_results(self):
    metrics_keys = list(self.history.history.keys())
    # summarize history for accuracy
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Acccuracy')
    ax1.set_ylim(0, 1) 
    ax1.plot(self.history.history[metrics_keys[1]], label='train')
    ax1.plot(self.history.history[metrics_keys[3]], label='test')
    ax1.set_label(ax1.legend(loc='lower right'))
    
    # summarize history for loss
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylim(0, max([max(self.history.history[metrics_keys[1]]), max(self.history.history[metrics_keys[0]])]))
    ax2.plot(self.history.history[metrics_keys[0]], label='train')
    ax2.plot(self.history.history[metrics_keys[2]], label='test')
    ax2.set_label(ax2.legend(loc='lower right'))
    
    plt.show()
  
  def save_model(self):
    is_to_save = input("Do you want save the model [y/n]: ")
    if is_to_save == 'y':
      self.neural_network_model.save('mlp_model.h5')
      print('Model saved in mlp_model.h5 file')
      return
    print('The current model was discarded!')
    return

  def evaluate_model(self, model):
    model_loaded = tf.keras.models.load_model(model)
    loss_value, accuracy_value = model_loaded.evaluate(self.x_test, self.test_label)
    print("Loss value=", loss_value, "Accuracy value = {:5.2f}%" .format(100 * accuracy_value))
    predictions = model_loaded.predict(self.x_test)

    y_pred = []
    for i in range(len(predictions)):
      if predictions[i] > 0.5:
        y_pred.append(1)
      else:
        y_pred.append(0)
    
    y_true = self.y_test.T.tolist()[0]
    print('Predict: ', y_pred)
    print('True:    ', y_true)

    # ground truth on vertical
    print('---> Confusion Matrix <---')
    print(confusion_matrix(y_true, y_pred))
    print('--------------------------')

    # get accuracy comparing y_true with y_pred
    m = tf.compat.v1.keras.metrics.Accuracy()
    m.update_state(y_true, y_pred)
    print('Accuracy value directly: {:5.2f}%' .format(m.result().numpy() * 100))

  def _split_columns(self, data):
    # Dataframe hold ID, 9 attributes and 1 class attribute (label)
    x_index = data.iloc[:,[0]]
    x = data.iloc[:,list(range(1, self._number_of_attributes))]
    y = data.iloc[:,[self._label_column]]
    return x_index, x, y

  def _get_optimizer_from_name(self, name, lr):
    optimizer_dic = {
      'SGD': tf.keras.optimizers.SGD(lr=lr, momentum=0.5, nesterov=True),
      'RMSprop': tf.keras.optimizers.RMSprop(lr=lr),
      'Adagrad': tf.keras.optimizers.Adagrad(lr=lr),
      'Adadelta': tf.keras.optimizers.Adadelta(lr=lr),
      'Adam': tf.keras.optimizers.Adam(lr=lr),
      'Adamax': tf.keras.optimizers.Adamax(lr=lr),
      'Nadam': tf.keras.optimizers.Nadam(lr=lr)
    }
    return optimizer_dic[name]

if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  print(type(sys.argv[1]))
  if sys.argv[1] != 'train_model' and sys.argv[1] != 'evaluate_model':
    print('Input argument <' + sys.argv[1] + '> is invalid! Options are: train_model ou evaluate_model')
    sys.exit()

  mlp = MLP()
  mlp.get_data('dataset/')
  mlp.prepare_data()

  if sys.argv[1] == 'train_model':
    with open('hyper_parameters.yaml', 'r') as config_file:
      hyper_parameter = yaml.load(config_file, Loader=yaml.FullLoader)

    mlp.create_model(hyper_parameter['hiden_layer_neurons'],
                     hyper_parameter['activation_functions'],
                     hyper_parameter['dropout_parameters'],
                     hyper_parameter['loss'],
                     hyper_parameter['metrics'],
                     hyper_parameter['optimizer'],
                     hyper_parameter['lr'])

    mlp.train(hyper_parameter['batch_size'],
              hyper_parameter['epochs'])

    mlp.show_results()
    mlp.save_model()
  # else:
  #   mlp.evaluate_model('mlp_model.h5')

import tensorflow as tf
import numpy as np

class ResidualUnit(tf.keras.Model):
      def __init__(self, filter_in, filter_out, kernel_size):
    super(ResidualUnit, self).__init__()
    # batch normalization -> ReLu -> Conv Layer
    # 여기서 ReLu 같은 경우는 변수가 없는 Layer이므로 여기서 굳이 initialize 해주지 않는다. (call쪽에서 사용하면 되므로)

    self.bn1 = tf.keras.layers.BatchNormalization()
    self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding="same")

    self.bn2 = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding="same")

    # identity를 어떻게 할지 정의
    # 원래 Residual Unit을 하려면 위의 순서로 진행한 뒤, 바로 X를 더해서 내보내면 되는데,
    # 이 X와 위의 과정을 통해 얻은 Feature map과 차원이 동일해야 더하기 연산이 가능할 것이므로
    # 즉, 위에서 filter_in과 filter_out이 같아야 한다는 의미이다.
    # 하지만, 다를 수 있으므로 아래와 같은 작업을 거친다.

    if filter_in == filter_out:
      self.identity = lambda x: x
    else:
      self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding="same")

  # 아래에서 batch normalization은 train할때와 inference할 때 사용하는 것이 달라지므로 옵션을 줄것이다.
  def call(self, x, training=False, mask=None):
    h = self.bn1(x, training=training)
    h = tf.nn.relu(h)
    h = self.conv1(h)

    h = self.bn2(h, training=training)
    h = tf.nn.relu(h)
    h = self.conv2(h)
    return self.identity(x) + h

class ResnetLayer(tf.keras.Model):
      # 아래 arg 중 filter_in : 처음 입력되는 filter 개수를 의미
  # Resnet Layer는 Residual unit이 여러개가 있게끔해주는것이므로
  # filters : [32, 32, 32, 32]는 32에서 32로 Residual unit이 연결되는 형태
  def __init__(self, filter_in, filters, kernel_size):
    super(ResnetLayer, self).__init__()
    self.sequnce = list()
    # [16] + [32, 32, 32]
    # 아래는 list의 length가 더 작은 것을 기준으로 zip이 되어서 돌아가기 때문에
    # 앞의 list의 마지막 element 32는 무시된다.
    # zip([16, 32, 32, 32], [32, 32, 32])
    for f_in, f_out in zip([filter_in] + list(filters), filters):
      self.sequnce.append(ResidualUnit(f_in, f_out, kernel_size))

  def call(self, x, training=False, mask=None):
    for unit in self.sequnce:
      # 위의 batch normalization에서 training이 쓰였기에 여기서 넘겨 주어야 한다.
      x = unit(x, training=training)
    return x

class ResNet(tf.keras.Model):
      def __init__(self):
    super(ResNet, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(8, (3,3), padding="same", activation="relu") # 28X28X8

    self.res1 = ResnetLayer(8, (16, 16), (3, 3)) # 28X28X16
    self.pool1 = tf.keras.layers.MaxPool2D((2,2)) # 14X14X16

    self.res2 = ResnetLayer(16, (32, 32), (3, 3)) # 14X14X32
    self.pool2 = tf.keras.layers.MaxPool2D((2,2)) # 7X7X32

    self.res3 = ResnetLayer(32, (64, 64), (3, 3)) # 7X7X64

    self.flatten = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(128, activation="relu")
    self.dense2 = tf.keras.layers.Dense(10, activation="softmax")

  def call(self, x, training=False, mask=None):
    x = self.conv1(x)

    x = self.res1(x, training=training)
    x = self.pool1(x)
    x = self.res2(x, training=training)
    x = self.pool2(x)
    x = self.res3(x, training=training)

    x = self.flatten(x)
    x = self.dense1(x)
    return self.dense2(x)

# Implement training loop
@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
  with tf.GradientTape() as tape:
    # training=True 꼭 넣어주기!!
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  train_loss(loss)
  train_accuracy(labels, predictions)

# Implement algorithm test
@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
  # training=False 꼭 넣어주기!!
  predictions = model(images, training=False)

  t_loss = loss_object(labels, predictions)
  test_loss(t_loss)
  test_accuracy(labels, predictions)
  

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype(np.float32)
x_test = x_test[..., tf.newaxis].astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 모델 생성
model = ResNet()

# 손실함수 정의 및 최적화 기법 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# 평가지표 정의
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for epoch in range(EPOCHS):
      for images, labels in train_ds:
    train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

  for test_images, test_labels in test_ds:
    test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

  template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))
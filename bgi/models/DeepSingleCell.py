import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import Dropout, BatchNormalization, Add
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from bgi.layers.attention import AttentionWithContext


class ExpandDimsLayer(Layer):
    def __init__(self, axis, name=None):
        super(ExpandDimsLayer, self).__init__(name=name)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({"axis": self.axis})
        return config


def multi_embedding_attention_transfer(
    supvised_train: bool = False,
    scan_train: bool = False,
    multi_max_features: list = [40000],
    mult_feature_names: list = ['Gene'],
    embedding_dims=128,
    num_classes=5,
    activation='softmax',
    head_1=128,
    head_2=128,
    head_3=128,
    drop_rate=0.05,
    include_attention: bool = False,
    use_bias=True,
):
    assert len(multi_max_features) == len(mult_feature_names)

    x_feature_inputs = []
    x_value_inputs = []
    features = []

    if include_attention:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            feature_input = Input(shape=(None,), name=f'Input-{name}-Feature')
            value_input = Input(shape=(None,), name=f'Input-{name}-Value', dtype='float32')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            embedding = Embedding(
                max_length, embedding_dims, input_length=None, name=f'{name}-Embedding'
            )(feature_input)

            sparse_value = ExpandDimsLayer(axis=2, name=f'{name}-Expand-Dims')(value_input)
            sparse_value = BatchNormalization(name=f'{name}-BN-1')(sparse_value)
            x = tf.multiply(embedding, sparse_value, name=f'{name}-Multiply')

            weight_output, a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))
            x = BatchNormalization(name=f'{name}-BN-3')(x)

            features.append(x)

        inputs = x_feature_inputs + x_value_inputs

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            value_input = Input(shape=(max_length,), name=f'Input-{name}-Value', dtype='float32')
            x_value_inputs.append(value_input)

            sparse_value = BatchNormalization(name=f'{name}-BN-1')(value_input)
            x = Dense(head_1, name=f'{name}-Projection-0', activation='relu')(sparse_value)
            x = BatchNormalization(name=f'{name}-BN-3')(x)
            features.append(x)

        inputs = x_value_inputs

    if len(features) > 1:
        feature = Add()(features)
    else:
        feature = features[0]

    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='Projection-1', activation='relu')(dropout)

    return tf.keras.Model(inputs=inputs, outputs=output)


class EncoderHead(tf.keras.Model):
    def __init__(self, hidden_size=128, dropout=0.05):
        super(EncoderHead, self).__init__()
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        feature_output = self.feature_bn1(feature_output)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc2(feature_output)

        return feature_output

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
import tensorflow_addons as tfa
from utils.layers import Mil_Attention

class RBFKernelFn(tf.keras.layers.Layer):
    """
    RBF kernel for Gaussian processes.
    """
    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)
        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0.0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0.0),
            dtype=dtype,
            name='length_scale')

    def call(self, x):
        # Never called -- this is just a layer so it can hold variables
        # in a way Keras understands.
        return x

    @property
    def kernel(self):
        # tf.print('amp', tf.nn.softplus(0.1 * self._amplitude))
        # tf.print('ls', tf.nn.softplus(10.0 * self._length_scale))
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude), # 0.1
            length_scale=tf.nn.softplus(10.0 * self._length_scale) # 5.
        )


def build_model(config, data_dims, num_training_points):
    num_inducing_points = config['model']['inducing_points']
    num_classes = config['data']['num_classes']
    optimizer = config['model']['optimizer']
    learning_rate = config['model']['learning_rate']
    mc_samples = 20

    def mc_sampling(x):
        """
        Monte Carlo Sampling of the GP output distribution.
        :param x:
        :return:
        """
        samples = x.sample(mc_samples)
        return samples


    def mc_integration(x):
        """
        Monte Carlo integration is basically replacing an integral with the mean of samples.
        Here we take the mean of the previously generated samples.
        :param x:
        :return:
        """
        x = tf.math.reduce_mean(x, axis=1)
        out = tf.reshape(x, [num_classes])
        return out

    def custom_softmax(x):
        x = tf.reshape(x, shape=[mc_samples, 1, -1])
        # x = tf.reshape(x, shape=[1, -1])
        x = tf.keras.activations.softmax(x, axis=-1)
        out = tf.reshape(x, shape=[mc_samples, -1])
        # out = tf.reshape(x, shape=[-1])
        return out

    def attention_multiplication(i):
        # a = tf.ones_like(i[0])
        a = i[0]
        f = i[1]
        # tf.print('attention', a)
        # tf.print('features', f)
        if config['model']['attention'] == 'gp':
            out = tf.linalg.matvec(f, a, transpose_a=True)
        else:
            a = tf.reshape(a, shape=[-1])
            out = tf.linalg.matvec(f, a, transpose_a=True)
            out = tf.reshape(out, [1, f.shape[1]])
        return out

    def att_mean(x):
        dims = x.shape[1]
        x = tf.reduce_mean(x, axis=0)
        out = tf.reshape(x, [1, dims])
        return out

    def reshape_pre_mc_integration(x):
        out = tf.reshape(x, shape=[1, mc_samples, num_classes])
        return out

    def reshape_final(x):
        out = tf.reshape(x, shape=[num_classes])
        return out

    input = tf.keras.layers.Input(shape=data_dims)
    if config['model']['hidden_layer_size_0'] == 0:
        f = input
    else:
        f = tf.keras.layers.Dense(config['model']['hidden_layer_size_0'], activation='relu')(input)

    if config['model']['hidden_layer_size_1'] != 0:
        f = tf.keras.layers.Dense(config['model']['hidden_layer_size_1'], activation='relu')(f)

    if config['model']['hidden_layer_size_2'] != 0:
        f = tf.keras.layers.Dense(config['model']['hidden_layer_size_2'], activation='relu')(f)

    if config['model']['hidden_layer_size_att'] == 0:
        x = f
    else:
        x = tf.keras.layers.Dense(config['model']['hidden_layer_size_att'], activation=config['model']['hidden_layer_act_att'])(f)

    att_type = config['model']['attention']

    if att_type == 'gp':
        if config['model']['att_sigmoid']:
            x = tf.keras.layers.Activation('sigmoid')(x)
        # x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tfp.layers.VariationalGaussianProcess(
            mean_fn=lambda x: tf.ones([1]) * config['model']['prior'],
            num_inducing_points=num_inducing_points,
            kernel_provider=RBFKernelFn(),
            event_shape=[1],  # output dimensions
            inducing_index_points_initializer=tf.keras.initializers.RandomUniform(
                minval=0.3, maxval=0.7, seed=None
            ),
            jitter=10e-3,
            convert_to_tensor_fn=tfp.distributions.Distribution.sample,
            variational_inducing_observations_scale_initializer=tf.initializers.constant(
                0.01 * np.tile(np.eye(num_inducing_points, num_inducing_points), (1, 1, 1))),
            )(x)

        x = tf.keras.layers.Lambda(mc_sampling, name='instance_attention')(x)
        a = tf.keras.layers.Lambda(custom_softmax, name='instance_softmax')(x)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])

        x = tf.reshape(x, shape=[mc_samples, 1, -1])
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax_a')(x)
        x = tf.keras.layers.Lambda(reshape_pre_mc_integration, name='bag_softmax')(x)
        output = tf.keras.layers.Lambda(mc_integration)(x)
    elif att_type == 'deterministic' or att_type == 'deterministic_gated':
        if att_type == 'deterministic':
            gated = False
        elif att_type == 'deterministic_gated':
            gated = True
        a = Mil_Attention(f.shape[1], output_dim=0, name='instance_attention', use_gated=gated)(x)
        x = tf.keras.layers.Lambda(attention_multiplication)([a, f])
        # x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax_a')(x)
        output = tf.keras.layers.Lambda(reshape_final)(x)
    elif att_type == 'baseline_mean':
        x = tf.keras.layers.Lambda(att_mean, name='instance_attention')(x)
        x = tf.keras.layers.Dense(num_classes, activation='softmax', name='bag_softmax_a')(x)
        output = tf.keras.layers.Lambda(reshape_final)(x)


    model = tf.keras.Model(inputs=input, outputs=output, name="sgp_mil")
    instance_model = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('instance_attention').output)

    if config['model']['attention'] == 'gp':
        bag_level_uncertainty_model =  tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('bag_softmax').output)
        model.add_loss(kl_loss(model, num_training_points, config['model']['kl_factor']))
    else:
        bag_level_uncertainty_model = None

    if num_classes == 2:
        loss  = tf.keras.losses.CategoricalCrossentropy()
        metrics =  [tf.keras.metrics.CategoricalAccuracy()]
    else:
        # loss = tf.keras.losses.SparseCategoricalCrossentropy()
        # metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

    if optimizer == 'sgd':
        opt = tf.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
    else:
        print('Choose valid optimizer')

    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics)

    return model, instance_model, bag_level_uncertainty_model

def kl_loss(head, num_training_points, kl_factor):
    # tf.print('kl_div: ', kl_div)
    num_training_points = tf.constant(num_training_points, dtype=tf.float32)

    layer_name = 'variational_gaussian_process'
    vgp_layer = head.get_layer(layer_name)

    def _kl_loss():
        kl_weight = tf.cast(kl_factor / num_training_points, tf.float32)
        kl_div = tf.reduce_sum(vgp_layer.submodules[5].surrogate_posterior_kl_divergence_prior())

        loss = tf.multiply(kl_weight, kl_div)
        return loss

    return _kl_loss
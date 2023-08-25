"""Various tensorflow quality-of-life utilities."""

import tensorflow as tf

# @tf.function
def array_diff(a, n=1):
    """Nth-order difference without axis specification."""
    # Check shape in the static and dynamic cases.
    if None not in a.get_shape().as_list():
        a_size = 1
        for s in a.get_shape().as_list():
            a_size *= s
        if n >= a_size:
            raise ValueError('n must be less than the size of array a.')
    else:
        with tf.control_dependencies([a]):
            tf.assert_less(n, tf.size(a),
                           message='n must be less than the size of array a.')

    # Flatten to 1-D array.
    a = tf.reshape(a, [-1])
    return a[n:] - a[:-n]


def trapezoidal_integration(x, y):
    """Performs numerical univariate integration via the trapazoidal rule."""
    # Check shapes in the static and dynamic cases.
    if (None not in x.get_shape().as_list() and
        None not in y.get_shape().as_list()):
        x_shape = x.get_shape().as_list()
        y_shape = y.get_shape().as_list()
        if len(x_shape) != len(y_shape) or x_shape != y_shape:
            raise ValueError('x and y must have the same shape.')
    else:
        with tf.control_dependencies([x, y]):
            tf.assert_equal(tf.shape(x), tf.shape(y),
                            message='x and y must have the same shape.')

    # Flatten to 1-D arrays.
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    # https://en.wikipedia.org/wiki/Trapezoidal_rule#Numerical_implementation
    dx = tf.cast(array_diff(x), dtype=y.dtype)
    return tf.reduce_sum(0.5 * (y[:-1] + y[1:]) * dx)
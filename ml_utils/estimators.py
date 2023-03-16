import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


__all__ = ["get_flops", "time_per_layer"]


def get_flops(model: tf.keras.Model, table=False) -> tuple[float, float]:
    """
    Calculates the FLOPS (Floating Point Operations Per Second) and
        MACC (Multiply-Accumulate Operations) of a TensorFlow Keras model.

    Supported Layers:
    ---------
    Conv2D, DepthwiseConv2D, SeparableConv2D, Activation, BatchNormalization, InputLayer,
        Reshape, Add, Maximum, Concatenate, Average, pool, Flatten, Global Pooling,

    Args:
    ---------
        model (tf.keras.Model): TensorFlow Keras model
        table (bool, optional): If table is True, it prints a formatted table of the FLOPS and MACC values for each layer in the model.
            Defaults to False.

    Returns:
    ---------
        tuple[float, float]: Return the tuple of FLOPS and MACC

    Example:
    ---------
    ```
    import tensorflow as tf
    from ml_utils.estimator import get_flops, time_per_layer

    model = tf.keras.applications.ResNet50(
        weights=None, include_top=True, pooling=None, input_shape=(224, 224, 3)
    )
    get_flops(model, table=True)
    ```
    """
    if table is True:
        print(
            "%25s | %16s | %16s | %16s | %16s | %6s | %6s"
            % (
                "Layer Name",
                "Input Shape",
                "Output Shape",
                "Kernel Size",
                "Filters",
                "Strides",
                "FLOPS",
            )
        )
        print("-" * 170)

    t_flops = 0
    t_macc = 0
    
    # 1G
    factor = 1_000_000_000

    for layer in model.layers:
        name = layer.name
        o_shape, i_shape, strides, ks, filters = (
            ["", "", ""],
            ["", "", ""],
            [1, 1],
            [0, 0],
            [0, 0],
        )
        flops = 0
        macc = 0

        if "InputLayer" in str(layer):
            i_shape = layer.input.get_shape()[1:4].as_list()
            o_shape = i_shape

        elif "Reshape" in str(layer):
            i_shape = layer.input.get_shape()[1:4].as_list()
            o_shape = layer.output.get_shape()[1:4].as_list()

        elif (
            "Add" in str(layer)
            or "Maximum" in str(layer)
            or "Concatenate" in str(layer)
        ):
            i_shape = layer.input[0].get_shape()[1:4].as_list() + [len(layer.input)]
            o_shape = layer.output.get_shape()[1:4].as_list()
            flops = (len(layer.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]

        elif "Average" in str(layer) and "pool" not in str(layer):
            i_shape = layer.input[0].get_shape()[1:4].as_list() + [len(layer.input)]
            o_shape = layer.output.get_shape()[1:4].as_list()
            flops = len(layer.input) * i_shape[0] * i_shape[1] * i_shape[2]

        elif (
            "BatchNormalization" in str(layer)
            or "Activation" in str(layer)
            or "activation" in str(layer)
        ):
            i_shape = layer.input.get_shape()[1:4].as_list()
            o_shape = layer.output.get_shape()[1:4].as_list()
            bflops = 1
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor

        elif "pool" in str(layer) and ("Global" not in str(layer)):
            i_shape = layer.input.get_shape()[1:4].as_list()
            strides = layer.strides
            ks = layer.pool_size
            flops = (
                (i_shape[0] / strides[0])
                * (i_shape[1] / strides[1])
                * (ks[0] * ks[1] * i_shape[2])
            )

        elif "Flatten" in str(layer):
            i_shape = layer.input.shape[1:4].as_list()
            flops = 1
            out_vec = 1
            for i in range(len(i_shape)):
                flops *= i_shape[i]
                out_vec *= i_shape[i]
            o_shape = flops
            flops = 0

        elif "Dense" in str(layer):
            i_shape = layer.input.shape[1:4].as_list()[0]
            if i_shape == None:
                i_shape = out_vec

            o_shape = layer.output.shape[1:4].as_list()
            flops = 2 * (o_shape[0] * i_shape)
            macc = flops / 2

        elif "Padding" in str(layer):
            flops = 0

        elif "Global" in str(layer):
            i_shape = layer.input.get_shape()[1:4].as_list()
            flops = (i_shape[0]) * (i_shape[1]) * (i_shape[2])
            o_shape = [layer.output.get_shape()[1:4].as_list(), 1, 1]
            out_vec = o_shape

        elif (
            "Conv2D " in str(layer)
            and "DepthwiseConv2D" not in str(layer)
            and "SeparableConv2D" not in str(layer)
        ):
            strides = layer.strides
            ks = layer.kernel_size
            filters = layer.filters
            i_shape = layer.input.get_shape()[1:4].as_list()
            o_shape = layer.output.get_shape()[1:4].as_list()

            if filters == None:
                filters = i_shape[2]
            flops = 2 * (
                (filters * ks[0] * ks[1] * i_shape[2])
                * ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]))
            )
            macc = flops / 2

        elif (
            "Conv2D " in str(layer)
            and "DepthwiseConv2D" in str(layer)
            and "SeparableConv2D" not in str(layer)
        ):
            strides = layer.strides
            ks = layer.kernel_size
            filters = layer.filters
            i_shape = layer.input.get_shape()[1:4].as_list()
            o_shape = layer.output.get_shape()[1:4].as_list()

            if filters == None:
                filters = i_shape[2]
            flops = 2 * (
                (ks[0] * ks[1] * i_shape[2])
                * ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]))
            )
            macc = flops / 2

        t_macc += macc
        t_flops += flops

        if table is True:
            print(
                "%25s | %16s | %16s | %16s | %16s | %6s | %5.4f"
                % (
                    name,
                    str(i_shape),
                    str(o_shape),
                    str(ks),
                    str(filters),
                    str(strides),
                    flops,
                )
            )

    t_flops = t_flops / factor
    t_macc = t_macc / factor

    print()
    print("Total FLOPs: {:10.8f}".format(t_flops))
    print("Total MACCs: {:10.8f}".format(t_macc))
    print()
    
    return t_flops, t_macc


def time_per_layer(
    model: tf.keras.Model, visualize=False, figure: str = None
) -> np.ndarray[np.float32]:
    """
    Measures the time taken for each layer in the neural network model to process an input of ones.

    Args:
    ---------
        model (tf.keras.Model): TensorFlow Keras model
        visualize (bool, optional): A bar chart of the processing time for each layer. Defaults to False.
        save_figure (str, optional): Figure name to be saved in local. Defaults to None. No figure to be saved.

    Returns:
    ---------
        np.ndarray[np.float32]: Times of layers. NumPy array of dimensions (number of layers x 2).

    Example:
    ---------
    ```
    import tensorflow as tf
    from ml_utils.estimator import get_flops, time_per_layer

    model = tf.keras.applications.ResNet50(
        weights=None, include_top=True, pooling=None, input_shape=(224, 224, 3)
    )
    time_per_layer(model, visualize=True, figure="example.png")
    ```
    """
    times = np.zeros((len(model.layers), 2))
    inp = np.ones(model.input.shape[1:])

    for i in range(1, len(model.layers)):
        new_model = tf.keras.models.Model(
            inputs=[model.input], outputs=[model.layers[-i].output]
        )
        new_model.predict(inp[None, :, :, :])

        t_s = time.time()
        new_model.predict(inp[None, :, :, :])
        t_e2 = time.time() - t_s

        times[i, 1] = t_e2
        del new_model

    for i in range(0, len(model.layers) - 1):
        times[i, 0] = abs(times[i + 1, 1] - times[i, 1])

    times[-1, 0] = times[-1, 1]

    if visualize is True or figure is not None:
        plt.style.use("ggplot")
        x = [model.layers[-i].name for i in range(1, len(model.layers))]
        g = [times[i, 0] for i in range(1, len(times))]
        x_pos = np.arange(len(x))
        plt.bar(x, g, color="#7ed6df")
        plt.xlabel("Layers")
        plt.ylabel("Processing Time")
        plt.title("Processing Time of each Layer")
        plt.xticks(x_pos, x, rotation=45)
        if figure is not None:
            plt.savefig(figure)
        if visualize is True:
            plt.show()

    return times

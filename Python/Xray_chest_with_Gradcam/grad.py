from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2


class GradCAM:
    '''
    Class that creates heatmap of predicted photo.
    '''
    def __init__(self, model, classIdx, layerName=None):
        '''

        :param model: model of architecture
        :param classIdx: index of image
        :param layerName: initialize function to find last layer
        '''
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        '''
        Find last layer of neural network.
        :return: name of last layer
        '''
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

    def compute_heatmap(self, image, eps=1e-8):
        '''
        Compute heatmap of image.
        :param image: pass image that we want to create of heatmap on it.
        :param eps: constant for gradient
        :return: heatmap of image
        '''
        # using previously found layers to create model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        # Tensorflow gradient tape
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32) # change type to float32
            (convOutputs, predictions) = gradModel(inputs)  # go through our model with gradient
            loss = predictions[:, self.classIdx]    # prediction of given photo
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1]) # resize output to match our main model output
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_CIVIDIS):
        '''
        Aplly hetamap on image.
        :param heatmap: heatmap
        :param image: passed image
        :param alpha: alpha for color
        :param colormap: set of colors for gradient.
        :return: heatmap and image with heatmap.
        '''
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)
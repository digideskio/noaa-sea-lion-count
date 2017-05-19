"""
Module to train and use neural networks.
"""
import os
import abc
import functools
import settings
import keras
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
import scipy.misc
import skimage.transform
import numpy as np
import metrics

PRETRAINED_MODELS = {
    "vgg16":     VGG16,
    "vgg19":     VGG19,
    "inception": InceptionV3,
    "xception":  Xception,
    "resnet":    ResNet50
}

class Learning:
    def __init__(self, data_type = "original", input_shape = (300,300,3), prediction_class_type = "multi", mini_batch_size = 32, tensor_board = True, validate = True):
        """
        :param prediction_class_type: "single" or "multi"
        """

        self.model = None
        self.prediction_class_type = prediction_class_type
        self.mini_batch_size = mini_batch_size
        self.tensor_board = tensor_board
        self.validate = validate

        logger.info("Starting...")
        loader = data.Loader()
        transform = data.LoadTransformer(data.AugmentationTransformer(next = data.ResizeTransformer(input_shape)))
        
        if data_type == 'original':
            train_data = loader.load_original_images()
        elif data_type == 'sea_lion_crops':
            throw(NotImplemented('Cropping still has to be implemented.'))
        elif data_type == 'region_crops':
            throw(NotImplemented('Cropping still has to be implemented.'))
        
        if validate:
            train_val_split = loader.train_val_split(train_data)
            self.iterator = data.DataIterator(train_val_split['train'], transform, batch_size = mini_batch_size, shuffle = True, seed = 42)
            self.val_iterator = data.DataIterator(train_val_split['test'], transform, batch_size = mini_batch_size, shuffle = True, seed = 42)
        else:
            self.iterator = data.DataIterator(train_data, transform, batch_size = mini_batch_size, shuffle = True, seed = 42)

    @abc.abstractmethod
    def build(self):
        throw(NotImplemented("Must be implemented by child class."))

    def train(self, epochs, weights_name):
        """
        Train the (previously loaded/set) model. It is assumed that the `model` object
        has been already configured (such as which layers of it are frozen and which not)
        
        :param epochs: the number of training epochs
        :param mini_batch_size: size of the mini batches
        :param weights_name: name for the h5py weights file to be written in the output folder
        """
        
        callbacks_list = []

        # Create weight output dir if it does not exist
        if not os.path.exists(settings.WEIGHTS_DIR):
            os.makedirs(settings.WEIGHTS_DIR)

        # Save the model with best validation accuracy during training
        weights_name = weights_name + ".e{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5"
        weights_path = os.path.join(settings.WEIGHTS_OUTPUT_DIR, weights_name)
        checkpoint = keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor = 'val_loss',
            verbose=1,
            save_best_only = True,
            mode = 'min')
        callbacks_list.append(checkpoint)
                 
        if self.tensor_board:
            # Output tensor board logs
            tf_logs = keras.callbacks.TensorBoard(
                log_dir = settings.TENSORBOARD_LOGS_DIR,
                histogram_freq = 1,
                write_graph = True,
                write_images = True)
            callbacks_list.append(tf_logs)

        # Train
        self.model.fit_generator(
            generator = self.iterator,
            steps_per_epoch = int(949/self.mini_batch_size), 
            epochs = epochs,
            validation_data = self.val_iterator if self.validate else None,
            validation_steps = int(0.3*949/self.mini_batch_size) if self.validate else None,
            workers = 2,
            callbacks = callbacks_list)


class TransferLearning(Learning):
	
    def __init__(self, *args, **kwargs):
        """
        TransferLearning initialization.
        """
        super().__init__(*args, **kwargs)

        self.base_model = None
        self.base_model_name = None
        self.model_name = None

    def extend(self):
        """
        Extend the model by stacking new (dense) layers on top of the network
        """
        x = self.base_model.output
        # x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(5, activation='softmax')(x)

        # This is the model we will train:
        self.model = keras.models.Model(input=self.base_model.input, output=predictions)

    def build(self, base_model_name, input_shape = None, extended_model_name = None, summary = False):
        """
        Build an extended model. A base model is first loaded disregarding its last layers and afterwards
        some new layers are stacked on top so the resulting model would be applicable to the
        fishering-monitoring problem
        
        :param base_model_name: model name to load and use as base model (`"vgg16"`,`"vgg19"`,`"inception"`,`"xception"`,`"resnet"`).
        :param input_shape: optional shape tuple (see input_shape of argument of base network used in Keras).
        :param extended_model_name: name for the extended model. It will affect only to the weights file to write on disk
        :param summary: whether to print the summary of the extended model
        """

        # Set the base model configuration and extended model name
        self.base_model_name = base_model_name
        self.base_model = PRETRAINED_MODELS[self.base_model_name](weights = 'imagenet', include_top = False, input_shape = input_shape)
        if not extended_model_name:
            extended_model_name = 'ext_' + base_model_name

        self.model_name = extended_model_name

        # Extend the base model
        print("Building %s using %s as the base model..." % (self.model_name, self.base_model_name))
        self.extend()
        print("Done building the model.")

        if summary:
            print(self.model.summary())
    
    def fine_tune_extended(self, epochs, input_weights_name, n_layers = 126):
        """
        Fine-tunes the extended model. It is assumed that the top part of the classifier has already been trained
        using the `train_top` method. It retrains the top part of the extended model and also some of the last layers
        of the base model with a low learning rate.
        
        :param epochs: the number of training epochs
        :param mini_batch_size: size of the mini batches
        :param input_weights_name: name of the h5py weights file to be loaded as start point (output of `train_top`).
        :param n_layers: freeze every layer from the bottom of the extended model until the nth layer. Default is
        126 which is reasonable for the Xception model
        """

        # Load weights
        self.model.load_weights(os.path.join(settings.WEIGHTS_DIR,input_weights_name))

        # Freeze layers
        for layer in self.model.layers[:n_layers]:
           layer.trainable = False

        for layer in self.model.layers[n_layers:]:
           layer.trainable = True

        if self.prediction_class_type == "single":
            loss = "binary_crossentropy"
            metrics_ = ['accuracy', metrics.precision, metrics.recall]
        elif self.prediction_class_type == "multi":
            loss = "categorical_crossentropy"
            metrics_ = ['accuracy']

        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss=loss, metrics = metrics_)
        weights_name = self.model_name+'.finetuned'
        
        # Train
        self.train(epochs, weights_name)
        
    def train_top(self, epochs):
        """
        Trains the top part of the extended model. In other words it trains the extended model but freezing every
        layer of the base model.
        
        :param epochs: training epochs
        :param mini_batch_size: size of the mini batches
        """
    
        # Freeze all convolutional base_model layers
        for layer in self.base_model.layers:
            layer.trainable = False
            
        if self.prediction_class_type == "single":
            loss = "binary_crossentropy"
            metrics_ = ['accuracy', metrics.precision, metrics.recall]
        elif self.prediction_class_type == "multi":
            loss = "categorical_crossentropy"
            metrics_ = ['accuracy']

        self.model.compile(
                optimizer = 'adam',
                loss = loss,
                metrics = metrics_)
                
        weights_name = self.model_name+'.toptrained'
        
        # Train
        self.train(epochs, weights_name)

class TransferLearningSeaLionOrNoSeaLion(TransferLearning):

    def __init__(self, *args, **kwargs):
        """
        TransferLearningSeaLionOrNoSeaLion initialization.
        """
        super().__init__(*args, **kwargs)


    def extend(self):
        """
        Extend the model by stacking new (dense) layers on top of the network
        """
        x = self.base_model.output
        #x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Flatten()(x)
        #x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(1, activation='sigmoid')(x)

        # This is the model we will train:
        self.model = keras.models.Model(input=self.base_model.input, output=predictions)

class LearningFullyConvolutional(TransferLearning):

    def __init__(self, *args, **kwargs):
        """
        LearningFullyConvolutional initialization.
        """
        super().__init__(*args, **kwargs)

    def extend(self, w, b, num_classes):
        """
        Extend the model by stacking new (dense) layers on top of the network
        """

        x = self.base_model.layers[-2].output

        # A 1x1 convolution, with the same number of output channels as there are classes
        fullyconv = keras.layers.Convolution2D(num_classes, 1, 1, name="fullyconv")(x)

        output = keras.layers.Activation("sigmoid")(fullyconv)

        # This is fully convolutional model:
        self.model = keras.models.Model(input=self.base_model.input, output=output)

        if num_classes > 1:
            last_layer = self.model.layers[-2]
        else:
            last_layer = self.model.layers[-2]

        print("Loaded weight shape:", w.shape)
        print("Last conv layer weights shape:", last_layer.get_weights()[0].shape)

        # Set weights of fullyconv layer:
        w_reshaped = w.reshape((1, 1, 2048, num_classes))

        last_layer.set_weights([w_reshaped, b])

    def build(self, weights_file = None, num_classes = 1000):
        if weights_file == None:
            self.base_model = ResNet50(include_top = False, weights = 'imagenet')

            # Load weights of the regular last dense layer of ResNet50
            import h5py
            h5f = h5py.File(os.path.join(settings.IMAGENET_DIR, 'resnet_weights_dense.h5'),'r')
            w = h5f['w'][:]
            b = h5f['b'][:]
            h5f.close()
        else:
            # Get the trained model
            trained_model = keras.models.load_model(os.path.join(settings.WEIGHTS_DIR, weights_file), custom_objects={'precision': metrics.precision, 'recall': metrics.recall})
            print(trained_model.summary())
            
            # Get the base model (i.e., without the last dense layer)
            trained_base_model = keras.models.Model(input=trained_model.input, output=trained_model.layers[-3].output)

            # Get the bare (untrained) ResNet50 architecture and load in the trained model's weights
            resnet = ResNet50(include_top = False, weights = None)
            resnet.set_weights(trained_base_model.get_weights())

            # Set it as the base model
            self.base_model = resnet

            # Get the weights of the trained model's last dense layer
            weights = trained_model.layers[-1].get_weights()
            w = weights[0]
            b = weights[1]

        self.extend(w, b, num_classes)

    def forward_pass_resize(self, img, img_size):
        from keras.applications.imagenet_utils import preprocess_input

        img_raw = img
        print("img shape before resizing: %s" % (img_raw.shape,))

        # Resize
        img = scipy.misc.imresize(img_raw, size=img_size).astype("float32")

        # Add axis
        img = img[np.newaxis]

        # Preprocess for use in imagenet        
        img = preprocess_input(img)

        print("img batch size shape before forward pass:", img.shape)
        z = self.model.predict(img)

        return z

    def build_heatmap(self, img, img_size, axes):
        probas = self.forward_pass_resize(img, img_size)

        import imagenettool

        x = probas[0, :, :, np.array(axes)].sum(axis=0)
        print("size of heatmap: " + str(x.shape))
        return x

    def build_multi_scale_heatmap(self, img, axes = [0]):

        shape = img.shape

        heatmaps = []
        
        for scale in [2.0, 1.75, 1.25, 1.0]:
            size = (round(shape[0] * scale), round(shape[1] * scale), shape[2])
            heatmaps.append(self.build_heatmap(img, size, axes))

        largest_heatmap_shape = heatmaps[0].shape

        heatmaps = [skimage.transform.resize(heatmap, largest_heatmap_shape, preserve_range = True).astype("float32") for heatmap in heatmaps]
        geom_avg_heatmap = np.power(functools.reduce(lambda x, y: x*y, heatmaps), 1.0 / len(heatmaps))
        
        return geom_avg_heatmap

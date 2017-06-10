"""
Module to train and use neural networks.
"""
import os
import abc
import data
import functools
import keras
import metrics
import numpy as np
import math
import scipy.misc
import settings
import skimage.transform
from time import strftime

PRETRAINED_MODELS = {
    "vgg16":     keras.applications.VGG16,
    "vgg19":     keras.applications.VGG19,
    "inception": keras.applications.InceptionV3,
    "xception":  keras.applications.Xception, # TensorFlow ONLY
    "resnet":    keras.applications.ResNet50
}

class Learning:
    def __init__(self, data_type = "original", input_shape = (300,300,3), prediction_class_type = "multi", class_balancing = True, mini_batch_size = 32, tensor_board = False, validate = True):
        """
        :param prediction_class_type: "single" or "multi"
        """

        self.model = None
        self.input_shape = input_shape
        self.prediction_class_type = prediction_class_type
        self.mini_batch_size = mini_batch_size
        self.tensor_board = tensor_board
        self.validate = validate
        self.data_type = data_type

        settings.logger.info("Starting...")
        loader = data.Loader()
        transform = self.data_transformer()
        
        if data_type == 'original_train':
            train_data = loader.load_original_images(dataset = 'train')
        if data_type == 'original_test':
            train_data = loader.load_original_images(dataset = 'test_st1')
        else:
            train_data = loader.load_crop_images(data_type = data_type)
            
        if data_type == 'heatmap_crops' and class_balancing:
            settings.logger.error('Class balancing can\'t be activated while data_type='+data_type)
        
        if validate:
            train_val_split = loader.train_val_split(train_data)
            self.iterator = data.DataIterator(train_val_split['train'], transform, batch_size = mini_batch_size, shuffle = True, seed = 42, class_balancing = class_balancing, class_transformation = self.data_class_transform())
            # No class balancing for validation
            self.val_iterator = data.DataIterator(train_val_split['validate'], transform, batch_size = mini_batch_size, shuffle = True, seed = 42, class_balancing = False, class_transformation = self.data_class_transform())

        else:
            self.iterator = data.DataIterator(train_data, transform, batch_size = mini_batch_size, shuffle = True, seed = 42, class_balancing = class_balancing, class_transformation = self.data_class_transform())
        
        
    def data_transformer(self):
        if self.input_shape == (224,224,3):
            #If input shape is (224, 224, 3) there is no need to use ResizeTransformer()
            settings.logger.info("Resizing images deactivated")
            if self.data_type == 'heatmap_crops':
                transformer = data.LoadTransformer(data.SyncedAugmentationTransformer())
            else:
                transformer = data.LoadTransformer(data.AugmentationTransformer())
        else:
            settings.logger.info("Resizing images activated")
            if self.data_type == 'heatmap_crops':
                transformer = data.LoadTransformer(data.SyncedAugmentationTransformer(next = data.ResizeTransformer(self.input_shape)))
            else:
                transformer = data.LoadTransformer(data.AugmentationTransformer(next = data.ResizeTransformer(self.input_shape)))
        return transformer
    
    def data_class_transform(self):
        return lambda x: x

    @abc.abstractmethod
    def build(self):
        throw(NotImplemented("Must be implemented by child class."))

    def train(self, epochs):
        """
        Train the (previously loaded/set) model. It is assumed that the `model` object
        has been already configured (such as which layers of it are frozen and which not)
        
        :param epochs: the number of training epochs
        :param mini_batch_size: size of the mini batches
        :param weights_name: name for the h5py weights file to be written in the output folder
        """
        
        callbacks_list = []
        trainable_layers = sum([int(layer.trainable) for layer in self.model.layers])
        # Create weight output dir if it does not exist
        base_name = self.arch_name + "-lay"+str(trainable_layers)

        # Save the model with best validation accuracy during training
        
        weights_name = base_name+"-ep{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5"
        
        weights_path = os.path.join(settings.WEIGHTS_DIR, weights_name)
        checkpoint = keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor = 'val_loss',
            verbose=1,
            save_best_only = True,
            mode = 'min')
        callbacks_list.append(checkpoint)
                 
        if self.tensor_board:
            log_dir = os.path.join(settings.TENSORBOARD_LOGS_DIR,strftime("%Y%m%dT%H%M%S")+'_'+base_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            # Output tensor board logs
            tf_logs = keras.callbacks.TensorBoard(
                log_dir = log_dir,
                histogram_freq = 1,
                write_graph = True,
                write_images = True)
            callbacks_list.append(tf_logs)
            
        #TODO get unqie_instances automatically 
        unique_instances = 50000
        # Train
        steps_per_epoch = math.ceil(0.7*unique_instances/self.mini_batch_size)
        validation_steps = math.ceil(0.3*unique_instances/self.mini_batch_size) if self.validate else None
        self.print_layers_info()
        self.model.fit_generator(
            generator = self.iterator,
            steps_per_epoch = steps_per_epoch, 
            epochs = epochs,
            validation_data = self.val_iterator if self.validate else None,
            validation_steps = validation_steps,
            workers = 8,
            callbacks = callbacks_list)


        

class TransferLearning(Learning):
	
    def __init__(self, *args, **kwargs):
        """
        TransferLearning initialization.
        """
        super().__init__(*args, **kwargs)

        self.base_model = None
        self.model = None
        self.arch_name = None
        
    def print_layers_info(self):
        """
        Prints information about current frozen (non trainable) and unfrozen (trainable)
        layers
        """
        print(len(self.model.layers),'total layers (',len(self.base_model.layers),\
            'pretrained and',len(self.model.layers)-len(self.base_model.layers),'new stacked on top)')
        trainable = [layer.trainable for layer in self.model.layers]
        non_trainable = [not i for i in trainable]
        tr_pos = list(np.where(trainable)[0])
        nontr_pos = list(np.where(non_trainable)[0])
        if len(nontr_pos) > 0:
            print('\t',sum(non_trainable),'non-trainable layers: from',nontr_pos[0],'to',nontr_pos[-1])
        print('\t',sum(trainable),'trainable layers: from',tr_pos[0],'to',tr_pos[-1])
        print('Trainable layer map:',''.join([str(int(l.trainable)) for l in self.model.layers]))
        
    def unfreeze_last_pretrained_layers(self, n_layers = None, percentage = None):
        '''
        Un freeze some of the last pretrained layers of the model
        '''
        assert n_layers or percentage
        if percentage:
            assert percentage < 1
            n_layers = int(float(len(self.base_model.layers))*percentage)
        #print('Freezing last',n_layers,'of the pretrained model',self.arch,'...')
        for layer in self.base_model.layers[-n_layers:]:
            layer.trainable = True
            
    def freeze_all_pretrained_layers(self):
        '''
        Freeze all the pretrained layers. Note: a "pretrained layer" is named as such
        even after fine-tunning it
        '''
        settings.logger.info('Freezing all pretrained layers...')
        for layer in self.base_model.layers:
            layer.trainable = False
            
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

    def build(self, arch_name, input_shape = None):
        """
        Build an extended model. A base model is first loaded disregarding its last layers and afterwards
        some new layers are stacked on top so the resulting model would be applicable to the
        fishering-monitoring problem
        
        :param arch_name: model name to load and use as base model (`"vgg16"`,`"vgg19"`,`"inception"`,`"xception"`,`"resnet"`).
        :param input_shape: optional shape tuple (see input_shape of argument of base network used in Keras).
        :param summary: whether to print the summary of the extended model
        """

        # Set the base model configuration and extended model name
        self.arch_name = arch_name
        self.base_model = PRETRAINED_MODELS[self.arch_name](weights = 'imagenet', include_top = False, input_shape = input_shape)
        
        # Extend the base model
        settings.logger.info("Building network using %s as the pretrained architecture..." % (self.arch_name))
        self.extend()
            
    def load_weights(self,input_weights_name):
        weights_filepath = os.path.join(settings.WEIGHTS_DIR,input_weights_name)
        self.model.load_weights(weights_filepath)
        settings.logger.info("Loaded weights "+weights_filepath)
    
    def fine_tune_extended(self, epochs, input_weights_name, n_layers = 126, perc = None):
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
        self.load_weights(input_weights_name)
        if not perc:
            # Freeze layers
            for layer in self.model.layers[:n_layers]:
               layer.trainable = False

            for layer in self.model.layers[n_layers:]:
               layer.trainable = True
        else:
            self.freeze_all_pretrained_layers()
            self.unfreeze_last_pretrained_layers(percentage = perc)

        if self.prediction_class_type == "single":
            loss = "binary_crossentropy"
            metrics_ = ['accuracy', metrics.precision, metrics.recall]
        elif self.prediction_class_type == "multi":
            loss = "categorical_crossentropy"
            metrics_ = ['accuracy']

        self.model.compile(optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss=loss, metrics = metrics_)
        #weights_name = self.model_name+'.finetuned'
        
        # Train
        self.train(epochs)    
        

        
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
        elif self.prediction_class_type == 'odm':
            loss = 'mean_squared_error'
            metrics_ = ['accuracy']

        self.model.compile(
                optimizer = 'adam',
                loss = loss,
                metrics = metrics_)
                
        #weights_name = self.model_name+'.toptrained'
        
        # Train
        self.train(epochs)

class TransferLearningSeaLionHeatmap(TransferLearning):
    '''
    This class should be used for training on heatmaps.    
    '''

    def __init__(self, *args, **kwargs):
        """
        TransferLearningSeaLionHeatmap initialization.
        """
        super().__init__(*args, **kwargs)


    def extend(self):
        """
        Extend the model by stacking new (dense) layers on top of the network
        """
        if self.arch_name != 'xception':
            raise Exception("TransferLearningSeaLionHeatmap is only available with the Xception architecture, not "+self.arch_name)
        
        x = self.base_model.output
        x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((3, 3))(x)
        #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        #x = UpSampling2D((3, 3))(x)
        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        #The output shape of the predictions layer must match the shape of the obm (100x100x1, 80x80x1...)
        predictions = keras.layers.Conv2D(filters = 1,
                                         kernel_size = (3, 3),
                                         activation='sigmoid',
                                         padding='same')(x)
        self.model = keras.models.Model(input=self.base_model.input, output=predictions)
        settings.logger.info("Output shape of last layer is "+str(self.model.layers[-1].output_shape))
        #print(self.model.summary())
class TransferLearningSeaLionOrNoSeaLion(TransferLearning):
    '''
    This class should be used for "Sea Lion or no Sea Lion" network and
    for "Herd or no Herd" network.    
    '''

    def __init__(self, *args, **kwargs):
        """
        TransferLearningSeaLionOrNoSeaLion initialization.
        """
        super().__init__(*args, **kwargs)

    def data_class_transform(self):
        return data.sea_lion_type_to_sea_lion_or_not

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

        settings.logger.info("Loaded weight shape:", w.shape)
        settings.logger.info("Last conv layer weights shape:", last_layer.get_weights()[0].shape)

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
            #print(trained_model.summary())
            
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
        settings.logger.info("img shape before resizing: %s" % (img_raw.shape,))

        # Resize
        img = scipy.misc.imresize(img_raw, size=img_size).astype("float32")

        # Add axis
        img = img[np.newaxis]

        # Preprocess for use in imagenet        
        img = preprocess_input(img)

        settings.logger.info("img batch size shape before forward pass:", img.shape)
        z = self.model.predict(img)

        return z

    def build_heatmap(self, img, img_size, axes):
        probas = self.forward_pass_resize(img, img_size)

        x = probas[0, :, :, np.array(axes)].sum(axis=0)
        settings.logger.info("size of heatmap: " + str(x.shape))
        return x

    def build_multi_scale_heatmap(self, img, scales=[1], axes = [0]): # scales=[1.5,1,0.7]

        shape = img.shape

        heatmaps = []
        
        for scale in scales:
            size = (round(shape[0] * scale), round(shape[1] * scale), shape[2])
            heatmaps.append(self.build_heatmap(img, size, axes))

        largest_heatmap_shape = heatmaps[0].shape

        heatmaps = [skimage.transform.resize(heatmap, largest_heatmap_shape, preserve_range = True).astype("float32") for heatmap in heatmaps]
        geom_avg_heatmap = np.power(functools.reduce(lambda x, y: x*y, heatmaps), 1.0 / len(heatmaps))
        #avg_heatmap = functools.reduce(lambda x, y: x+y, heatmaps) / len(heatmaps)
        
        return geom_avg_heatmap


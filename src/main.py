"""
Main module
"""

from clize import run, parameters
import data
import settings

logger = settings.logger.getChild('main')

def test_iterators():
    logger.info('Starting...')

    loader = data.Loader()
    train_data = loader.load_original_images()
    train_val_split = loader.train_val_split(train_data)

    transform = data.LoadTransformer(data.AugmentationTransformer(next = data.ResizeTransformer(settings.TRANSFORMATION_RESIZE_TO)))
    iterator = data.DataIterator(train_val_split['train'], transform, batch_size = 8, shuffle = True, seed = 42)

    batch = next(iterator)

    print('First in batch: {0}'.format(batch[0].shape))
    print('Batch size: {0}'.format(len(batch)))

def generate_crops(total_crops:int, crop_type:parameters.one_of('sealion_crops', 'region_crops')):
    if crop_type == 'sealion_crops':
        throw(NotImplemented('Cropping still has to be implemented.'))
    crops_per_round = 100
    rounds = round(total_crops / crops_per_round)
    logger.info('Attempting to make '+str(total_crops)+' region crops in '+str(rounds)+' rounds...')
    for i in range(rounds):
        cropper = data.Cropper(crop_size = 224, total_crops = crops_per_round, pos_perc = 0.5, min_sealions_herd = 10)
        cropper.find_crops()
        cropper.save_crops()

# "network": (# layers frozen in finetuning, network file to continue with)
# numbers taken from previous project, might need to be changed
NETWORKS = {
    'vgg16':     (0,   'insert-vgg16-network-weights-file-here.hdf5'),
    'vgg19':     (17,  'insert-vgg19-network-weights-file-here.hdf5'),
    'inception': (125, 'insert-inception-network-weights-file-here.hdf5'), # no numbers for this one
    'xception':  (125, 'insert-xception-network-weights-file-here.hdf5'),
    'resnet':    (75,  'insert-resnet-network-weights-file-here.hdf5')
}

def train_top_network(task:parameters.one_of('binary', 'type'), network:parameters.one_of(*sorted(NETWORKS.keys())), data_type:parameters.one_of('original', 'sealion_crops', 'region_crops')):
    """
    Train the top dense layer of an extended network.
    
    task: the task to train for ("binary" for sea lion or not, "type" for sea lion type)
    
    network: the network architecture to train (VGG16, VGG19, Inception, XCeption, ResNet)
    
    data_type: which data to use as training/validation set ("original", "sealion_crops", "region_crops")
    """

    from network import TransferLearning, TransferLearningSeaLionOrNoSeaLion
    
    if data_type == 'original':
        input_shape = settings.TRANSFORMATION_RESIZE_TO
    elif data_type == 'sea_lion_crops':
        input_shape = (100,100,3)
    elif data_type == 'region_crops':
        input_shape = (300,300,3)

    if task == 'type':
        tl = TransferLearning(data_type = data_type, input_shape = input_shape, prediction_class_type = "multi", mini_batch_size=16)
    elif task == 'binary':
        tl = TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = "single", mini_batch_size=16)

    tl.build(network.lower(), input_shape = input_shape, summary = False)
    tl.train_top(epochs = 100)

def fine_tune_network(task:parameters.one_of('binary', 'type'), network:parameters.one_of(*sorted(NETWORKS.keys())), data_type:parameters.one_of('original', 'sealion_crops', 'region_crops')):
    """
    Fine-tune a trained extended network. To do this, first the top
    of the extended network must have been trained.
    
    task: the task to train for ("binary" for sea lion or not, "type" for sea lion type)
    
    network: the network architecture to train (VGG16, VGG19, Inception, XCeption, ResNet)
    
    data_type: which data to use as training/validation set ("original", "sealion_crops", "region_crops")
    """

    from network import TransferLearning, TransferLearningSeaLionOrNoSeaLion

    if data_type == 'original':
        input_shape = settings.TRANSFORMATION_RESIZE_TO
    elif data_type == 'sea_lion_crops':
        input_shape = (100,100,3)
    elif data_type == 'region_crops':
        input_shape = (300,300,3)
    
    if task == 'type':
        tl = TransferLearning(data_type = data_type, input_shape = input_shape, prediction_class_type = "multi", mini_batch_size=16)
    elif task == 'binary':
        tl = TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = "single", mini_batch_size=16)

    tl.build(network.lower(), input_shape = input_shape, summary = False)
    tl.fine_tune_extended(
        epochs = 100,
        input_weights_name = NETWORKS[network.lower()][1],
        n_layers = NETWORKS[network.lower()][0])





if __name__ == '__main__':
    run(test_iterators,
        generate_crops,
        train_top_network,
        fine_tune_network)

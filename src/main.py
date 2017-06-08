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
    train_data = loader.load_crop_images(data_type = 'region_crops')
    #train_data = loader.load_original_images()
    train_val_split = loader.train_val_split(train_data)

    #transform = data.LoadTransformer(data.AugmentationTransformer(next = data.ResizeTransformer(settings.TRANSFORMATION_RESIZE_TO)))
    transform = data.LoadTransformer(data.AugmentationTransformer(next = data.ResizeTransformer((224,224,3))))
    #transform = data.AugmentationTransformer(next = data.ResizeTransformer((224,224,3)))
    iterator = data.DataIterator(train_val_split['train'], transform, batch_size = 16, shuffle = True, seed = 42)
    
    for i in range(10):
        batch = next(iterator)
        print('First in batch: {0}'.format(batch[0].shape))
        print('Batch size: {0}'.format(len(batch)))
        
    
def generate_region_crops(min_sealions_pos:int, blackout:bool):
    #python3 main.py generate-region-crops 5 True
    import cropping
    
    output_size = 224
    crop_size = 400
    diameter = 100
    cropper = cropping.RegionCropper(crop_size = crop_size, attention = blackout, diameter = diameter, output_size = output_size)

    sliding_perc = 0.1
    max_overlap_perc = 0.5
    plot = False
    cropper.find_pos_crops_dataset(min_sealions_pos, max_overlap_perc, plot, sliding_perc)

    wanted_crops = 70000
    max_sealions_neg = 0
    cropper.find_neg_crops_dataset(wanted_crops, max_sealions_neg)

def generate_individual_crops(num_negative_crops:int, *, sea_lion_size=100, ignore_pups=False, blackout=False, blackout_diameter=100):
    #python3 main.py generate-individual-crops 20000
    """
    Create positive and negative crops of sea lions from the original training data.
    
    num_negative_crops: the total number of negative crops generated over the whole data set
    
    sea_lion_size: the width and height of a sea lion crop in the image (always square)
    
    ignore_pups: if true, no pups are included in the data set (neither in positive nor negative samples)
    
    blackout: if true, the crops' corners will be made black and only a circle will remain

    blackout_diameter: the diameter of the blackout mask (only used if blackout = True)
    """
    import cropping
    cropping.generate_individual_crops(sea_lion_size, num_negative_crops, ignore_pups, blackout, blackout_diameter)

def generate_overlap_masks():
    #python3 main.py generate-overlap-masks
    """
    Generate boolean masks for the training set's black regions,
    in which False indicates that a pixel is black (overlaps with another image).
    Do this before generating crops.
    """
    
    import cropping
    cropping.generate_overlap_masks()

def generate_heatmaps(dataset:parameters.one_of('train', 'test_st1'), network_type:parameters.one_of('region', 'individual')):
    """
    Generate fully convolutional heatmaps 
    
    dataset: the data set to apply it to ('train' or 'test_st1')
    
    network_type: type of network to apply it to ('region' or 'individual')
    """
    import heatmap
    heatmap.generate_heatmaps(dataset, network_type)


# "network": (# layers frozen in finetuning, network file to continue with)
# numbers taken from previous project, might need to be changed
NETWORKS = {
    'vgg16':     (0,   'vgg16-lay2-ep018-tloss0.5491-vloss0.5485.hdf5'),
    'vgg19':     (17,  'insert-vgg19-network-weights-file-here.hdf5'),
    'inception': (125, 'insert-inception-network-weights-file-here.hdf5'), # no numbers for this one
    'xception':  (125, 'xception-lay2-ep001-tloss0.9953-vloss0.8017.hdf5'),
    'resnet':    (75,  'resnet-lay2-ep016-tloss0.0832-vloss0.0521.hdf5')
}

def train_top_network(task:parameters.one_of('binary', 'type'), network:parameters.one_of(*sorted(NETWORKS.keys())), data_type:parameters.one_of('original', 'sea_lion_crops', 'region_crops')):
    #nice -19 python3 main.py train-top-network binary vgg16 region_crops
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
        input_shape = (197,197,3)
    elif data_type == 'region_crops':
        input_shape = (224,224,3)

    if task == 'type':
        tl = TransferLearning(data_type = data_type, input_shape = input_shape, prediction_class_type = "multi", mini_batch_size=16)
    elif task == 'binary':
        tl = TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = "single", mini_batch_size=64, tensor_board = True)

    tl.build(network.lower(), input_shape = input_shape, summary = False)
    tl.train_top(epochs = 200)

def fine_tune_network(task:parameters.one_of('binary', 'type'), network:parameters.one_of(*sorted(NETWORKS.keys())), data_type:parameters.one_of('original', 'sea_lion_crops', 'region_crops')):
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
        input_shape = (197,197,3)
    elif data_type == 'region_crops':
        input_shape = (224,224,3)
    
    if task == 'type':
        tl = TransferLearning(data_type = data_type, input_shape = input_shape, prediction_class_type = "multi", mini_batch_size=16)
    elif task == 'binary':
        tl = TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = "single", mini_batch_size=64)

    tl.build(network.lower(), input_shape = input_shape, summary = False)

    tl.fine_tune_extended(
        epochs = 200,
        input_weights_name = NETWORKS[network.lower()][1],
        n_layers = NETWORKS[network.lower()][0])

def fine_tune_network_perc(task:parameters.one_of('binary', 'type'), network:parameters.one_of(*sorted(NETWORKS.keys())), perc:float, data_type:parameters.one_of('original', 'sealion_crops', 'region_crops')):
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
        input_shape = (197,197,3)
    elif data_type == 'region_crops':
        input_shape = (224,224,3)
    
    if task == 'type':
        tl = TransferLearning(data_type = data_type, input_shape = input_shape, prediction_class_type = "multi", mini_batch_size=16)
    elif task == 'binary':
        tl = TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = "single", mini_batch_size=64)

    tl.build(network.lower(), input_shape = input_shape, summary = False)

    tl.fine_tune_extended(
        epochs = 200,
        input_weights_name = NETWORKS[network.lower()][1],
        perc = perc)

if __name__ == '__main__':
    run(test_iterators,
        generate_region_crops,
        generate_individual_crops,
        generate_overlap_masks,
        generate_heatmaps,
        train_top_network,
        fine_tune_network,
        fine_tune_network_perc)

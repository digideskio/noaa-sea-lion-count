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
        
def generate_positive_region_crops(crop_size:int, min_sealions_pos:int, max_overlap_perc:float, plot:bool, sliding_perc:float):
    #nice -19 python main.py generate-positive-region-crops 500 15 0.5 False 0.1
    import cropping
    cropper = cropping.RegionCropper(crop_size)
    cropper.find_pos_crops_dataset(min_sealions_pos, max_overlap_perc, plot, sliding_perc)     
def generate_negative_region_crops(crop_size:int, wanted_crops:int, max_sealions_neg:int):
    #nice -19 python main.py generate-negative-region-crops 500 1000 2
    import cropping
    cropper = cropping.RegionCropper(crop_size)
    cropper.find_neg_crops_dataset(wanted_crops, max_sealions_neg)

def generate_naive_region_crops(total_crops:int):
    #python main.py generate-naive-region-crops 200000
    """
    Divide the cropping task in rounds to leverage the trade-off between efficiency and risk (the Cropper class
    first find all the crops and then writes them to disk) 
    """
    import cropping

    crops_per_round = 1000
    rounds = round(total_crops / crops_per_round)
    logger.info('Attempting to make '+str(total_crops)+' region crops in '+str(rounds)+' rounds...')
    for i in range(rounds):
        cropper = cropping.RegionCropper(crop_size = 224, total_crops = crops_per_round, pos_perc = 0.5, min_sealions_herd = 10)
        cropper.find_crops()
        cropper.save_crops()

def generate_individual_crops(num_negative_crops:int, *, sea_lion_size=100, ignore_pups=False):
    #python3 main.py generate-individual-crops 20000
    """
    Create positive and negative crops of sea lions from the original training data.
    
    num_negative_crops: the total number of negative crops generated over the whole data set
    
    sea_lion_size: the width and height of a sea lion crop in the image (always square)
    
    ignore_pups: if true, no pups are included in the data set (neither in positive nor negative samples)
    """
    import cropping
    cropping.generate_individual_crops(sea_lion_size, num_negative_crops, ignore_pups)

def generate_overlap_masks():
    #python3 main.py generate-overlap-masks
    """
    Generate boolean masks for the training set's black regions,
    in which False indicates that a pixel is black (overlaps with another image).
    Do this before generating crops.
    """
    
    import cropping
    cropping.generate_overlap_masks()

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
        input_shape = (100,100,3)
    elif data_type == 'region_crops':
        input_shape = (224,224,3)

    if task == 'type':
        tl = TransferLearning(data_type = data_type, input_shape = input_shape, prediction_class_type = "multi", mini_batch_size=16)
    elif task == 'binary':
        tl = TransferLearningSeaLionOrNoSeaLion(data_type = data_type, input_shape = input_shape, prediction_class_type = "single", mini_batch_size=64)

    tl.build(network.lower(), input_shape = input_shape, summary = False)
    tl.train_top(epochs = 200)

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
        input_shape = (100,100,3)
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
        generate_positive_region_crops,
        generate_negative_region_crops,
        generate_individual_crops,
        fine_tune_network_perc,
        generate_overlap_masks,
        train_top_network,
        fine_tune_network)

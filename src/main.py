"""
Main module
"""

import settings
import data

logger = settings.logger.getChild("main")

def run():
    logger.info("Starting...")

    loader = data.Loader()
    train_data = loader.load_original_images()
    train_val_split = loader.train_val_split(train_data)

    transform = data.LoadTransformer(data.AugmentationTransformer(next = data.ResizeTransformer((2000,2000,3))))
    iterator = data.DataIterator(train_val_split['train'], transform, batch_size = 8, shuffle = True, seed = 42)

    batch = next(iterator)

    print("First in batch: %s" % batch[0]['m'])
    print("Batch size: %s" % len(batch))
    

if __name__ == "__main__":
    run()

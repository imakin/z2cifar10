import tensorflow as tf
import tensorflow_datasets as tfds

class Dataset():
    def __init__(self, train, test, val):
        self.train = train
        self.test = test
        self.val = val

class Datasets():
    def __init__(self):
        self.c10 = self.load_cifar10()
        self.c2 = self.load_cifar2()

    def load_cifar10(self):
        train_data = tfds.load(
            'cifar10', split='train[:90%]', as_supervised=True, shuffle_files=True
        )
        val_data = tfds.load(
            'cifar10', split='train[90%:]', as_supervised=True, shuffle_files=True
        )
        test_data = tfds.load(
            'cifar10', split='test', as_supervised=True, shuffle_files=True
        )
        train_dataset = self.preprocess_dataset(train_data)
        val_dataset = self.preprocess_dataset(val_data)
        test_dataset = self.preprocess_dataset(test_data)
        return Dataset(train_dataset, test_dataset, val_dataset)

    def load_cifar2(self):
        """
        dari 10 kelas menjadi 2 superclass (animal, vehicle)
        0: airplane
        1: automobile
        2: bird
        3: cat
        4: deer
        5: dog
        6: frog
        7: horse
        8: ship
        9: truck
        """
        animal_labels = [2, 3, 4, 5, 6]
        train_data = tfds.load(
            'cifar10', split='train[:90%]', as_supervised=True, shuffle_files=True
        )
        val_data = tfds.load(
            'cifar10', split='train[90%:]', as_supervised=True, shuffle_files=True
        )
        test_data = tfds.load(
            'cifar10', split='test', as_supervised=True, shuffle_files=True
        )
        train_dataset = self.preprocess_cifar2_dataset(train_data, animal_labels)
        val_dataset = self.preprocess_cifar2_dataset(val_data, animal_labels)
        test_dataset = self.preprocess_cifar2_dataset(test_data, animal_labels)
        return Dataset(train_dataset, test_dataset, val_dataset)

    def preprocess_dataset(self, ds):
        batch_dimension = 1024
        ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))
        return ds.batch(batch_dimension).prefetch(tf.data.experimental.AUTOTUNE)

    def preprocess_cifar2_dataset(self, ds, animal_labels):
        batch_dimension = 1024
        def map_fn(img, lbl):
            is_animal = tf.reduce_any([tf.equal(lbl, l) for l in animal_labels])
            lbl2 = tf.where(is_animal, 0, 1)
            return tf.cast(img, tf.float32) / 255.0, lbl2
        ds = ds.map(map_fn)
        return ds.batch(batch_dimension).prefetch(tf.data.experimental.AUTOTUNE)

datasets = Datasets()
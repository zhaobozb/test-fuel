import numpy
from collections import OrderedDict
import time


def test_iterable_dataset():
    from fuel.datasets import IterableDataset

    seed = 1234
    rng = numpy.random.RandomState(seed)
    features = rng.randint(256, size=(8, 2, 2))
    targets = rng.randint(4, size=(8, 1))

    dataset = IterableDataset(iterables=OrderedDict([('features', features), ('targets', targets)]),
                              axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                                       ('targets', ('batch', 'index'))]))

    print('Provided sources are {}.'.format(dataset.provides_sources))
    print('Sources are {}.'.format(dataset.sources))
    print('Axis labels are {}.'.format(dataset.axis_labels))
    print('Dataset contains {} examples.'.format(dataset.num_examples))

    state = dataset.open()
    while True:
        try:
            print(dataset.get_data(state=state))
        except StopIteration:
            print('Iteration over')
            break

    state = dataset.reset(state=state)
    print(dataset.get_data(state=state))

    dataset.close(state=state)


def test_indexabel_dataset():
    from fuel.datasets import IndexableDataset

    seed = 1234
    rng = numpy.random.RandomState(seed)
    features = rng.randint(256, size=(8, 2, 2))
    targets = rng.randint(4, size=(8, 1))

    dataset = IndexableDataset(indexables=OrderedDict([('features', features),
                                                       ('targets', targets)]),
                               axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                                        ('targets', ('batch', 'index'))]))

    state = dataset.open()
    print('State is {}.'.format(state))

    print(dataset.get_data(state=state, request=[1, 0]))

    dataset.close(state=state)


def test_iterate_scheme():
    from fuel.datasets import IndexableDataset
    from fuel.schemes import (SequentialScheme, ShuffledScheme,SequentialExampleScheme, ShuffledExampleScheme)

    seed = 1234
    rng = numpy.random.RandomState(seed)
    features = rng.randint(256, size=(8, 2, 2))
    targets = rng.randint(4, size=(8, 1))

    dataset = IndexableDataset(indexables=OrderedDict([('features', features),
                                                       ('targets', targets)]),
                               axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                                        ('targets', ('batch', 'index'))]))

    schemes = [SequentialScheme(examples=8, batch_size=5),
               ShuffledScheme(examples=8, batch_size=3),
               SequentialExampleScheme(examples=8),
               ShuffledExampleScheme(examples=8)]

    # for scheme in schemes:
    #     print(list(scheme.get_request_iterator()))

    state = dataset.open()
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=3)

    for request in scheme.get_request_iterator():
        data = dataset.get_data(state=state, request=request)
        print(data[0].shape, data[1].shape)

    dataset.close(state)


def test_data_stream():
    from fuel.datasets import IndexableDataset
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme

    seed = 1234
    rng = numpy.random.RandomState(seed)
    features = rng.randint(256, size=(8, 2, 2))
    targets = rng.randint(4, size=(8, 1))

    dataset = IndexableDataset(indexables=OrderedDict([('features', features),
                                                       ('targets', targets)]),
                               axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                                        ('targets', ('batch', 'index'))]))

    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=10)
    data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)
    for i, data in enumerate(data_stream.get_epoch_iterator()):
        print('epoch '+str(i), data[0].shape, data[1].shape)
        time.sleep(1)


def test_transformer():
    from fuel.transformers import ScaleAndShift
    from fuel.datasets import IndexableDataset
    from fuel.streams import DataStream
    from fuel.schemes import ShuffledScheme

    seed = 1234
    rng = numpy.random.RandomState(seed)
    features = rng.randint(256, size=(8, 2, 2))
    targets = rng.randint(4, size=(8, 1))

    dataset = IndexableDataset(indexables=OrderedDict([('features', features),
                                                       ('targets', targets)]),
                               axis_labels=OrderedDict([('features', ('batch', 'height', 'width')),
                                                        ('targets', ('batch', 'index'))]))
    scheme = ShuffledScheme(examples=dataset.num_examples, batch_size=2)
    data_stream = DataStream(dataset=dataset, iteration_scheme=scheme)

    scale = 1.0 / features.std()
    shift = - scale * features.mean()

    standardized_stream = ScaleAndShift(data_stream=data_stream,
                                        scale=scale, shift=shift,
                                        which_sources=('features',))

    for batch in standardized_stream.get_epoch_iterator():
        print(batch)


def gen_own_dataset_v1():
    import h5py

    train_vector_features = numpy.load('train_vector_features.npy')
    test_vector_features = numpy.load('test_vector_features.npy')
    train_image_features = numpy.load('train_image_features.npy')
    test_image_features = numpy.load('test_image_features.npy')
    train_targets = numpy.load('train_targets.npy')
    test_targets = numpy.load('test_targets.npy')

    f = h5py.File('dataset.hdf5', mode='w')
    vector_features = f.create_dataset('vector_features', (100, 10), dtype='float32')
    image_features = f.create_dataset('image_features', (100, 3, 5, 5), dtype='uint8')
    targets = f.create_dataset('targets', (100, 1), dtype='uint8')

    vector_features[...] = numpy.vstack([train_vector_features, test_vector_features])
    image_features[...] = numpy.vstack([train_image_features, test_image_features])
    targets[...] = numpy.vstack([train_targets, test_targets])

    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'feature'
    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'channel'
    image_features.dims[2].label = 'height'
    image_features.dims[3].label = 'width'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    split_array = numpy.empty(6, dtype=numpy.dtype([('split', 'a', 5),
                                                    ('source', 'a', 15),
                                                    ('start', numpy.int64, 1),
                                                    ('stop', numpy.int64, 1),
                                                    ('indices', h5py.special_dtype(ref=h5py.Reference)),
                                                    ('available', numpy.bool, 1),
                                                    ('comment', 'a', 1)]))
    split_array[0:3]['split'] = 'train'.encode('utf8')
    split_array[3:6]['split'] = 'test'.encode('utf8')
    split_array[0:6:3]['source'] = 'vector_features'.encode('utf8')
    split_array[1:6:3]['source'] = 'image_features'.encode('utf8')
    split_array[2:6:3]['source'] = 'targets'.encode('utf8')
    split_array[0:3]['start'] = 0
    split_array[0:3]['stop'] = 90
    split_array[3:6]['start'] = 90
    split_array[3:6]['stop'] = 100
    split_array[:]['indices'] = h5py.Reference()
    split_array[:]['available'] = True
    split_array[:]['comment'] = '.'.encode('utf8')
    f.attrs['split'] = split_array

    f.flush()
    f.close()


def gen_own_dataset_v2():
    import h5py
    from fuel.datasets.hdf5 import H5PYDataset

    train_vector_features = numpy.load('train_vector_features.npy')
    test_vector_features = numpy.load('test_vector_features.npy')
    train_image_features = numpy.load('train_image_features.npy')
    test_image_features = numpy.load('test_image_features.npy')
    train_targets = numpy.load('train_targets.npy')
    test_targets = numpy.load('test_targets.npy')

    f = h5py.File('dataset.hdf5', mode='w')
    vector_features = f.create_dataset('vector_features', (100, 10), dtype='float32')
    image_features = f.create_dataset('image_features', (100, 3, 5, 5), dtype='uint8')
    targets = f.create_dataset('targets', (100, 1), dtype='uint8')

    vector_features[...] = numpy.vstack([train_vector_features, test_vector_features])
    image_features[...] = numpy.vstack([train_image_features, test_image_features])
    targets[...] = numpy.vstack([train_targets, test_targets])

    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'feature'
    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'channel'
    image_features.dims[2].label = 'height'
    image_features.dims[3].label = 'width'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    split_dict = {'train': {'vector_features': (0, 90), 'image_features': (0, 90), 'targets': (0, 90)},
                  'test': {'vector_features': (90, 100), 'image_features': (90, 100), 'targets': (90, 100)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()


def gen_vlen_dataset():
    import h5py
    from fuel.datasets.hdf5 import H5PYDataset

    sizes = numpy.random.randint(3, 9, size=(100,))
    train_image_features = [numpy.random.randint(256, size=(3, size, size)).astype('uint8') for size in sizes[:90]]
    test_image_features = [numpy.random.randint(256, size=(3, size, size)).astype('uint8') for size in sizes[90:]]

    f = h5py.File('dataset_vlen.h5', mode='w')
    f['vector_features'] = numpy.vstack([numpy.load('train_vector_features.npy'), numpy.load('test_vector_features.npy')])
    f['targets'] = numpy.vstack([numpy.load('train_targets.npy'), numpy.load('test_targets.npy')])

    f['vector_features'].dims[0].label = 'batch'
    f['vector_features'].dims[1].label = 'feature'
    f['targets'].dims[0].label = 'batch'
    f['targets'].dims[1].label = 'index'

    all_image_features = train_image_features + test_image_features
    dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    image_features = f.create_dataset('image_features', (100,), dtype=dtype)
    image_features[...] = [image.flatten() for image in all_image_features]
    image_features.dims[0].label = 'batch'

    image_features_shapes = f.create_dataset('image_features_shapes', (100, 3), dtype='int32')
    image_features_shapes[...] = numpy.array([image.shape for image in all_image_features])

    image_features.dims.create_scale(image_features_shapes, 'shapes')
    image_features.dims[0].attach_scale(image_features_shapes)

    image_features_shape_labels = f.create_dataset('image_features_shape_labels', (3,), dtype='S7')
    image_features_shape_labels[...] = ['channel'.encode('utf8'), 'height'.encode('utf8'), 'width'.encode('utf8')]
    image_features.dims.create_scale(image_features_shape_labels, 'shape_labels')
    image_features.dims[0].attach_scale(image_features_shape_labels)

    split_dict = {'train': {'vector_features': (0, 90), 'image_features': (0, 90), 'targets': (0, 90)},
                  'test': {'vector_features': (90, 100), 'image_features': (90, 100), 'targets': (90, 100)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()

    train_set = H5PYDataset('dataset_vlen.h5', which_sets=('train',), sources=('image_features',))
    print(train_set.axis_labels['image_features'])
    handle = train_set.open()
    images, = train_set.get_data(handle, slice(0, 10))
    train_set.close(handle)
    print(images[0].shape, images[1].shape, images[2].shape, images[3].shape)


def play_h5():
    from fuel.datasets.hdf5 import H5PYDataset

    train_set = H5PYDataset('dataset.hdf5', which_sets=('train',))
    print(train_set.num_examples)
    print(train_set.provides_sources)
    print(train_set.axis_labels['image_features'])
    print(train_set.axis_labels['vector_features'])
    print(train_set.axis_labels['targets'])
    test_set = H5PYDataset('dataset.hdf5', which_sets=('test',))
    print(test_set.num_examples)

    handle = train_set.open()
    data = train_set.get_data(handle, slice(0, 10))
    print((data[0].shape, data[1].shape, data[2].shape))
    train_set.close(handle)



if __name__ == '__main__':
    # test_iterable_dataset()
    # test_indexabel_dataset()
    # test_iterate_scheme()
    # test_data_stream()
    # test_transformer()
    # gen_own_dataset_v2()
    # play_h5()
    gen_vlen_dataset()
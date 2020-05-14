import os


def imagenet_val_files_and_labels(dataset_directory):
    classes = open(os.path.join(dataset_directory, 'imagenet_lsvrc_2015_synsets.txt')).readlines()
    class_to_indx = {classes[i].split('\n')[0]: i for i in range(len(classes))}

    images_path = os.path.join(dataset_directory, 'val')
    filenames = []
    labels = []
    lines = open(os.path.join(dataset_directory, 'imagenet_2012_validation_synset_labels.txt'), 'r').readlines()
    for i, line in enumerate(lines):
        class_name = line.split('\n')[0]
        a = 'ILSVRC2012_val_%08d.JPEG' % (i + 1)
        filenames.append(f'{images_path}/{a}')
        labels.append(class_to_indx[class_name])
        # print(filenames[-1], labels[-1])

    return filenames, labels


def _find_classes(dir):
    # Faster and available in Python 3.5 and above
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
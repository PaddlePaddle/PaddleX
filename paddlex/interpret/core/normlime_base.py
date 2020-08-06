# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import numpy as np
import glob
import tqdm

from paddlex.interpret.as_data_reader.readers import read_image
import paddlex.utils.logging as logging
from . import lime_base
from ._session_preparation import compute_features_for_kmeans, gen_user_home
import paddlex.utils.logging as logging


def load_kmeans_model(fname):
    import pickle
    with open(fname, 'rb') as f:
        kmeans_model = pickle.load(f)

    return kmeans_model


def combine_normlime_and_lime(lime_weights, g_weights):
    pred_labels = lime_weights.keys()
    combined_weights = {y: [] for y in pred_labels}

    for y in pred_labels:
        normlized_lime_weights_y = lime_weights[y]
        lime_weights_dict = {
            tuple_w[0]: tuple_w[1]
            for tuple_w in normlized_lime_weights_y
        }

        normlized_g_weight_y = g_weights[y]
        normlime_weights_dict = {
            tuple_w[0]: tuple_w[1]
            for tuple_w in normlized_g_weight_y
        }

        combined_weights[y] = [
            (seg_k, lime_weights_dict[seg_k] * normlime_weights_dict[seg_k])
            for seg_k in lime_weights_dict.keys()
        ]

        combined_weights[y] = sorted(
            combined_weights[y], key=lambda x: np.abs(x[1]), reverse=True)

    return combined_weights


def avg_using_superpixels(features, segments):
    one_list = np.zeros((len(np.unique(segments)), features.shape[2]))
    for x in np.unique(segments):
        one_list[x] = np.mean(features[segments == x], axis=0)

    return one_list


def centroid_using_superpixels(features, segments):
    from skimage.measure import regionprops
    regions = regionprops(segments + 1)
    one_list = np.zeros((len(np.unique(segments)), features.shape[2]))
    for i, r in enumerate(regions):
        one_list[i] = features[int(r.centroid[0] + 0.5), int(r.centroid[1] +
                                                             0.5), :]
    return one_list


def get_feature_for_kmeans(feature_map, segments):
    from sklearn.preprocessing import normalize
    centroid_feature = centroid_using_superpixels(feature_map, segments)
    avg_feature = avg_using_superpixels(feature_map, segments)
    x = np.concatenate((centroid_feature, avg_feature), axis=-1)
    x = normalize(x)
    return x


def precompute_normlime_weights(list_data_,
                                predict_fn,
                                num_samples=3000,
                                batch_size=50,
                                save_dir='./tmp'):
    # save lime weights and kmeans cluster labels
    precompute_lime_weights(list_data_, predict_fn, num_samples, batch_size,
                            save_dir)

    # load precomputed results, compute normlime weights and save.
    fname_list = glob.glob(
        os.path.join(save_dir, 'lime_weights_s{}*.npy'.format(num_samples)))
    return compute_normlime_weights(fname_list, save_dir, num_samples)


def save_one_lime_predict_and_kmean_labels(lime_all_weights, image_pred_labels,
                                           cluster_labels, save_path):

    lime_weights = {}
    for label in image_pred_labels:
        lime_weights[label] = lime_all_weights[label]

    for_normlime_weights = {
        'lime_weights':
        lime_weights,  # a dict: class_label: (seg_label, weight)
        'cluster': cluster_labels  # a list with segments as indices.
    }

    np.save(save_path, for_normlime_weights)


def precompute_lime_weights(list_data_, predict_fn, num_samples, batch_size,
                            save_dir):
    root_path = gen_user_home()
    root_path = osp.join(root_path, '.paddlex')
    h_pre_models = osp.join(root_path, "pre_models")
    if not osp.exists(h_pre_models):
        if not osp.exists(root_path):
            os.makedirs(root_path)
        url = "https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz"
        pdx.utils.download_and_decompress(url, path=root_path)
    h_pre_models_kmeans = osp.join(h_pre_models, "kmeans_model.pkl")
    kmeans_model = load_kmeans_model(h_pre_models_kmeans)

    for data_index, each_data_ in enumerate(list_data_):
        if isinstance(each_data_, str):
            save_path = "lime_weights_s{}_{}.npy".format(
                num_samples, each_data_.split('/')[-1].split('.')[0])
            save_path = os.path.join(save_dir, save_path)
        else:
            save_path = "lime_weights_s{}_{}.npy".format(num_samples,
                                                         data_index)
            save_path = os.path.join(save_dir, save_path)

        if os.path.exists(save_path):
            logging.info(
                save_path + ' exists, not computing this one.', use_color=True)
            continue
        img_file_name = each_data_ if isinstance(each_data_,
                                                 str) else data_index
        logging.info(
            'processing ' + img_file_name + ' [{}/{}]'.format(data_index,
                                                              len(list_data_)),
            use_color=True)

        image_show = read_image(each_data_)
        result = predict_fn(image_show)
        result = result[0]  # only one image here.

        if abs(np.sum(result) - 1.0) > 1e-4:
            # softmax
            exp_result = np.exp(result)
            probability = exp_result / np.sum(exp_result)
        else:
            probability = result

        pred_label = np.argsort(probability)[::-1]

        # top_k = argmin(top_n) > threshold
        threshold = 0.05
        top_k = 0
        for l in pred_label:
            if probability[l] < threshold or top_k == 5:
                break
            top_k += 1

        if top_k == 0:
            top_k = 1

        pred_label = pred_label[:top_k]

        algo = lime_base.LimeImageInterpreter()
        interpreter = algo.interpret_instance(
            image_show[0],
            predict_fn,
            pred_label,
            0,
            num_samples=num_samples,
            batch_size=batch_size)

        X = get_feature_for_kmeans(
            compute_features_for_kmeans(image_show).transpose((1, 2, 0)),
            interpreter.segments)
        try:
            cluster_labels = kmeans_model.predict(X)
        except AttributeError:
            from sklearn.metrics import pairwise_distances_argmin_min
            cluster_labels, _ = pairwise_distances_argmin_min(
                X, kmeans_model.cluster_centers_)
        save_one_lime_predict_and_kmean_labels(
            interpreter.local_weights, pred_label, cluster_labels, save_path)


def compute_normlime_weights(a_list_lime_fnames, save_dir, lime_num_samples):
    normlime_weights_all_labels = {}

    for f in a_list_lime_fnames:
        try:
            lime_weights_and_cluster = np.load(f, allow_pickle=True).item()
            lime_weights = lime_weights_and_cluster['lime_weights']
            cluster = lime_weights_and_cluster['cluster']
        except:
            logging.info('When loading precomputed LIME result, skipping' +
                         str(f))
            continue
        logging.info('Loading precomputed LIME result,' + str(f))
        pred_labels = lime_weights.keys()
        for y in pred_labels:
            normlime_weights = normlime_weights_all_labels.get(y, {})
            w_f_y = [abs(w[1]) for w in lime_weights[y]]
            w_f_y_l1norm = sum(w_f_y)

            for w in lime_weights[y]:
                seg_label = w[0]
                weight = w[1] * w[1] / w_f_y_l1norm
                a = normlime_weights.get(cluster[seg_label], [])
                a.append(weight)
                normlime_weights[cluster[seg_label]] = a

            normlime_weights_all_labels[y] = normlime_weights

    # compute normlime
    for y in normlime_weights_all_labels:
        normlime_weights = normlime_weights_all_labels.get(y, {})
        for k in normlime_weights:
            normlime_weights[k] = sum(normlime_weights[k]) / len(
                normlime_weights[k])

    # check normlime
    if len(normlime_weights_all_labels.keys()) < max(
            normlime_weights_all_labels.keys()) + 1:
        logging.info(
            "\n" + \
            "Warning: !!! \n" + \
            "There are at least {} classes, ".format(max(normlime_weights_all_labels.keys()) + 1) + \
            "but the NormLIME has results of only {} classes. \n".format(len(normlime_weights_all_labels.keys())) + \
            "It may have cause unstable results in the later computation" + \
            " but can be improved by computing more test samples." + \
            "\n"
        )

    n = 0
    f_out = 'normlime_weights_s{}_samples_{}-{}.npy'.format(
        lime_num_samples, len(a_list_lime_fnames), n)
    while os.path.exists(os.path.join(save_dir, f_out)):
        n += 1
        f_out = 'normlime_weights_s{}_samples_{}-{}.npy'.format(
            lime_num_samples, len(a_list_lime_fnames), n)
        continue

    np.save(os.path.join(save_dir, f_out), normlime_weights_all_labels)
    return os.path.join(save_dir, f_out)


def precompute_global_classifier(dataset,
                                 predict_fn,
                                 save_path,
                                 batch_size=50,
                                 max_num_samples=1000):
    from sklearn.linear_model import LogisticRegression

    root_path = gen_user_home()
    root_path = osp.join(root_path, '.paddlex')
    h_pre_models = osp.join(root_path, "pre_models")
    if not osp.exists(h_pre_models):
        if not osp.exists(root_path):
            os.makedirs(root_path)
        url = "https://bj.bcebos.com/paddlex/interpret/pre_models.tar.gz"
        pdx.utils.download_and_decompress(url, path=root_path)
    h_pre_models_kmeans = osp.join(h_pre_models, "kmeans_model.pkl")
    kmeans_model = load_kmeans_model(h_pre_models_kmeans)

    image_list = []
    for item in dataset.file_list:
        image_list.append(item[0])

    x_data = []
    y_labels = []

    num_features = len(kmeans_model.cluster_centers_)

    logging.info(
        "Initialization for NormLIME: Computing each sample in the test list.",
        use_color=True)

    for each_data_ in tqdm.tqdm(image_list):
        x_data_i = np.zeros((num_features))
        image_show = read_image(each_data_)
        result = predict_fn(image_show)
        result = result[0]  # only one image here.
        c = compute_features_for_kmeans(image_show).transpose((1, 2, 0))

        segments = np.zeros((image_show.shape[1], image_show.shape[2]),
                            np.int32)
        num_blocks = 10
        height_per_i = segments.shape[0] // num_blocks + 1
        width_per_i = segments.shape[1] // num_blocks + 1

        for i in range(segments.shape[0]):
            for j in range(segments.shape[1]):
                segments[i,
                         j] = i // height_per_i * num_blocks + j // width_per_i

        # segments = quickshift(image_show[0], sigma=1)
        X = get_feature_for_kmeans(c, segments)

        try:
            cluster_labels = kmeans_model.predict(X)
        except AttributeError:
            from sklearn.metrics import pairwise_distances_argmin_min
            cluster_labels, _ = pairwise_distances_argmin_min(
                X, kmeans_model.cluster_centers_)

        for c in cluster_labels:
            x_data_i[c] = 1

        # x_data_i /= len(cluster_labels)

        pred_y_i = np.argmax(result)
        y_labels.append(pred_y_i)
        x_data.append(x_data_i)

    if len(np.unique(y_labels)) < 2:
        logging.info("Warning: The test samples in the dataset is limited.\n \
                     NormLIME may have no effect on the results.\n \
                     Try to add more test samples, or see the results of LIME.")
        num_classes = np.max(np.unique(y_labels)) + 1
        normlime_weights_all_labels = {}
        for class_index in range(num_classes):
            w = np.ones((num_features)) / num_features
            normlime_weights_all_labels[class_index] = {
                i: wi
                for i, wi in enumerate(w)
            }
        logging.info("Saving the computed normlime_weights in {}".format(
            save_path))

        np.save(save_path, normlime_weights_all_labels)
        return save_path

    clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    clf.fit(x_data, y_labels)

    num_classes = np.max(np.unique(y_labels)) + 1
    normlime_weights_all_labels = {}

    if len(y_labels) / len(np.unique(y_labels)) < 3:
        logging.info("Warning: The test samples in the dataset is limited.\n \
                     NormLIME may have no effect on the results.\n \
                     Try to add more test samples, or see the results of LIME.")

    if len(np.unique(y_labels)) == 2:
        # binary: clf.coef_ has shape of [1, num_features]
        for class_index in range(num_classes):
            if class_index not in clf.classes_:
                w = np.ones((num_features)) / num_features
                normlime_weights_all_labels[class_index] = {
                    i: wi
                    for i, wi in enumerate(w)
                }
                continue

            if clf.classes_[0] == class_index:
                w = -clf.coef_[0]
            else:
                w = clf.coef_[0]

            # softmax
            w = w - np.max(w)
            exp_w = np.exp(w * 10)
            w = exp_w / np.sum(exp_w)

            normlime_weights_all_labels[class_index] = {
                i: wi
                for i, wi in enumerate(w)
            }
    else:
        # clf.coef_ has shape of [len(np.unique(y_labels)), num_features]
        for class_index in range(num_classes):
            if class_index not in clf.classes_:
                w = np.ones((num_features)) / num_features
                normlime_weights_all_labels[class_index] = {
                    i: wi
                    for i, wi in enumerate(w)
                }
                continue

            coef_class_index = np.where(clf.classes_ == class_index)[0][0]
            w = clf.coef_[coef_class_index]

            # softmax
            w = w - np.max(w)
            exp_w = np.exp(w * 10)
            w = exp_w / np.sum(exp_w)

            normlime_weights_all_labels[class_index] = {
                i: wi
                for i, wi in enumerate(w)
            }

    logging.info("Saving the computed normlime_weights in {}".format(
        save_path))
    np.save(save_path, normlime_weights_all_labels)

    return save_path

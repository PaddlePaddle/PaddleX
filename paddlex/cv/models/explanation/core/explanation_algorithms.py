#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import numpy as np
import time

from . import lime_base
from ..as_data_reader.readers import read_image
from ._session_preparation import paddle_get_fc_weights, compute_features_for_kmeans, h_pre_models_kmeans
from .normlime_base import combine_normlime_and_lime, get_feature_for_kmeans, load_kmeans_model

import cv2


class CAM(object):
    def __init__(self, predict_fn, label_names):
        """

        Args:
            predict_fn: input: images_show [N, H, W, 3], RGB range(0, 255)
                        output: [
                        logits [N, num_classes],
                        feature map before global average pooling [N, num_channels, h_, w_]
                        ]

        """
        self.predict_fn = predict_fn
        self.label_names = label_names

    def preparation_cam(self, data_):
        image_show = read_image(data_)
        result = self.predict_fn(image_show)

        logit = result[0][0]
        if abs(np.sum(logit) - 1.0) > 1e-4:
            # softmax
            exp_result = np.exp(logit)
            probability = exp_result / np.sum(exp_result)
        else:
            probability = logit

        # only explain top 1
        pred_label = np.argsort(probability)
        pred_label = pred_label[-1:]

        self.predicted_label = pred_label[0]
        self.predicted_probability = probability[pred_label[0]]
        self.image = image_show[0]
        self.labels = pred_label

        fc_weights = paddle_get_fc_weights()
        feature_maps = result[1]
        
        l = pred_label[0]
        ln = l
        if self.label_names is not None:
            ln = self.label_names[l]

        print(f'predicted result: {ln} with probability {probability[pred_label[0]]:.3f}')
        return feature_maps, fc_weights

    def explain(self, data_, visualization=True, save_to_disk=True, save_outdir=None):
        feature_maps, fc_weights = self.preparation_cam(data_)
        cam = get_cam(self.image, feature_maps, fc_weights, self.predicted_label)

        if visualization or save_to_disk:
            import matplotlib.pyplot as plt
            from skimage.segmentation import mark_boundaries
            l = self.labels[0]
            ln = l 
            if self.label_names is not None:
                ln = self.label_names[l]

            psize = 5
            nrows = 1
            ncols = 2

            plt.close()
            f, axes = plt.subplots(nrows, ncols, figsize=(psize * ncols, psize * nrows))
            for ax in axes.ravel():
                ax.axis("off")
            axes = axes.ravel()
            axes[0].imshow(self.image)
            axes[0].set_title(f"label {ln}, proba: {self.predicted_probability: .3f}")

            axes[1].imshow(cam)
            axes[1].set_title("CAM")

        if save_to_disk and save_outdir is not None:
            os.makedirs(save_outdir, exist_ok=True)
            save_fig(data_, save_outdir, 'cam')

        if visualization:
            plt.show()

        return


class LIME(object):
    def __init__(self, predict_fn, label_names, num_samples=3000, batch_size=50):
        """
        LIME wrapper. See lime_base.py for the detailed LIME implementation.
        Args:
            predict_fn: from image [N, H, W, 3] to logits [N, num_classes], this is necessary for computing LIME.
            num_samples: the number of samples that LIME takes for fitting.
            batch_size: batch size for model inference each time.
        """
        self.num_samples = num_samples
        self.batch_size = batch_size

        self.predict_fn = predict_fn
        self.labels = None
        self.image = None
        self.lime_explainer = None
        self.label_names = label_names

    def preparation_lime(self, data_):
        image_show = read_image(data_)
        result = self.predict_fn(image_show)

        result = result[0]  # only one image here.

        if abs(np.sum(result) - 1.0) > 1e-4:
            # softmax
            exp_result = np.exp(result)
            probability = exp_result / np.sum(exp_result)
        else:
            probability = result

        # only explain top 1
        pred_label = np.argsort(probability)
        pred_label = pred_label[-1:]

        self.predicted_label = pred_label[0]
        self.predicted_probability = probability[pred_label[0]]
        self.image = image_show[0]
        self.labels = pred_label
        
        l = pred_label[0]
        ln = l
        if self.label_names is not None:
            ln = self.label_names[l]
            
        print(f'predicted result: {ln} with probability {probability[pred_label[0]]:.3f}')

        end = time.time()
        algo = lime_base.LimeImageExplainer()
        explainer = algo.explain_instance(self.image, self.predict_fn, self.labels, 0,
                                          num_samples=self.num_samples, batch_size=self.batch_size)
        self.lime_explainer = explainer
        print('lime time: ', time.time() - end, 's.')

    def explain(self, data_, visualization=True, save_to_disk=True, save_outdir=None):
        if self.lime_explainer is None:
            self.preparation_lime(data_)

        if visualization or save_to_disk:
            import matplotlib.pyplot as plt
            from skimage.segmentation import mark_boundaries
            l = self.labels[0]
            ln = l 
            if self.label_names is not None:
                ln = self.label_names[l]

            psize = 5
            nrows = 2
            weights_choices = [0.6, 0.7, 0.75, 0.8, 0.85]
            ncols = len(weights_choices)

            plt.close()
            f, axes = plt.subplots(nrows, ncols, figsize=(psize * ncols, psize * nrows))
            for ax in axes.ravel():
                ax.axis("off")
            axes = axes.ravel()
            axes[0].imshow(self.image)
            axes[0].set_title(f"label {ln}, proba: {self.predicted_probability: .3f}")

            axes[1].imshow(mark_boundaries(self.image, self.lime_explainer.segments))
            axes[1].set_title("superpixel segmentation")

            # LIME visualization
            for i, w in enumerate(weights_choices):
                num_to_show = auto_choose_num_features_to_show(self.lime_explainer, l, w)
                temp, mask = self.lime_explainer.get_image_and_mask(
                    l, positive_only=False, hide_rest=False, num_features=num_to_show
                )
                axes[ncols + i].imshow(mark_boundaries(temp, mask))
                axes[ncols + i].set_title(f"label {ln}, first {num_to_show} superpixels")

        if save_to_disk and save_outdir is not None:
            os.makedirs(save_outdir, exist_ok=True)
            save_fig(data_, save_outdir, 'lime', self.num_samples)

        if visualization:
            plt.show()

        return


class NormLIME(object):
    def __init__(self, predict_fn, label_names, num_samples=3000, batch_size=50,
                 kmeans_model_for_normlime=None, normlime_weights=None):
        if kmeans_model_for_normlime is None:
            try:
                self.kmeans_model = load_kmeans_model(h_pre_models_kmeans)
            except:
                raise ValueError("NormLIME needs the KMeans model, where we provided a default one in "
                                 "pre_models/kmeans_model.pkl.")
        else:
            print("Warning: It is *strongly* suggested to use the default KMeans model in pre_models/kmeans_model.pkl. "
                  "Use another one will change the final result.")
            self.kmeans_model = load_kmeans_model(kmeans_model_for_normlime)

        self.num_samples = num_samples
        self.batch_size = batch_size

        try:
            self.normlime_weights = np.load(normlime_weights, allow_pickle=True).item()
        except:
            self.normlime_weights = None
            print("Warning: not find the correct precomputed Normlime result.")

        self.predict_fn = predict_fn

        self.labels = None
        self.image = None
        self.label_names = label_names

    def predict_cluster_labels(self, feature_map, segments):
        return self.kmeans_model.predict(get_feature_for_kmeans(feature_map, segments))

    def predict_using_normlime_weights(self, pred_labels, predicted_cluster_labels):
        # global weights
        g_weights = {y: [] for y in pred_labels}
        for y in pred_labels:
            cluster_weights_y = self.normlime_weights.get(y, {})
            g_weights[y] = [
                (i, cluster_weights_y.get(k, 0.0)) for i, k in enumerate(predicted_cluster_labels)
            ]

            g_weights[y] = sorted(g_weights[y],
                                  key=lambda x: np.abs(x[1]), reverse=True)

        return g_weights

    def preparation_normlime(self, data_):
        self._lime = LIME(
            self.predict_fn,
            self.label_names,
            self.num_samples,
            self.batch_size
        )
        self._lime.preparation_lime(data_)

        image_show = read_image(data_)

        self.predicted_label = self._lime.predicted_label
        self.predicted_probability = self._lime.predicted_probability
        self.image = image_show[0]
        self.labels = self._lime.labels
        # print(f'predicted result: {self.predicted_label} with probability {self.predicted_probability: .3f}')
        print('performing NormLIME operations ...')

        cluster_labels = self.predict_cluster_labels(
            compute_features_for_kmeans(image_show).transpose((1, 2, 0)), self._lime.lime_explainer.segments
        )

        g_weights = self.predict_using_normlime_weights(self.labels, cluster_labels)

        return g_weights

    def explain(self, data_, visualization=True, save_to_disk=True, save_outdir=None):
        if self.normlime_weights is None:
            raise ValueError("Not find the correct precomputed NormLIME result. \n"
                             "\t Try to call compute_normlime_weights() first or load the correct path.")

        g_weights = self.preparation_normlime(data_)
        lime_weights = self._lime.lime_explainer.local_exp

        if visualization or save_to_disk:
            import matplotlib.pyplot as plt
            from skimage.segmentation import mark_boundaries
            l = self.labels[0]
            ln = l
            if self.label_names is not None:
                ln = self.label_names[l]

            psize = 5
            nrows = 4
            weights_choices = [0.6, 0.7, 0.75, 0.8, 0.85]
            nums_to_show = []
            ncols = len(weights_choices)

            plt.close()
            f, axes = plt.subplots(nrows, ncols, figsize=(psize * ncols, psize * nrows))
            for ax in axes.ravel():
                ax.axis("off")

            axes = axes.ravel()
            axes[0].imshow(self.image)
            axes[0].set_title(f"label {ln}, proba: {self.predicted_probability: .3f}")

            axes[1].imshow(mark_boundaries(self.image, self._lime.lime_explainer.segments))
            axes[1].set_title("superpixel segmentation")

            # LIME visualization
            for i, w in enumerate(weights_choices):
                num_to_show = auto_choose_num_features_to_show(self._lime.lime_explainer, l, w)
                nums_to_show.append(num_to_show)
                temp, mask = self._lime.lime_explainer.get_image_and_mask(
                    l, positive_only=False, hide_rest=False, num_features=num_to_show
                )
                axes[ncols + i].imshow(mark_boundaries(temp, mask))
                axes[ncols + i].set_title(f"LIME: first {num_to_show} superpixels")

            # NormLIME visualization
            self._lime.lime_explainer.local_exp = g_weights
            for i, num_to_show in enumerate(nums_to_show):
                temp, mask = self._lime.lime_explainer.get_image_and_mask(
                    l, positive_only=False, hide_rest=False, num_features=num_to_show
                )
                axes[ncols * 2 + i].imshow(mark_boundaries(temp, mask))
                axes[ncols * 2 + i].set_title(f"NormLIME: first {num_to_show} superpixels")

            # NormLIME*LIME visualization
            combined_weights = combine_normlime_and_lime(lime_weights, g_weights)
            self._lime.lime_explainer.local_exp = combined_weights
            for i, num_to_show in enumerate(nums_to_show):
                temp, mask = self._lime.lime_explainer.get_image_and_mask(
                    l, positive_only=False, hide_rest=False, num_features=num_to_show
                )
                axes[ncols * 3 + i].imshow(mark_boundaries(temp, mask))
                axes[ncols * 3 + i].set_title(f"Combined: first {num_to_show} superpixels")

            self._lime.lime_explainer.local_exp = lime_weights

        if save_to_disk and save_outdir is not None:
            os.makedirs(save_outdir, exist_ok=True)
            save_fig(data_, save_outdir, 'normlime', self.num_samples)

        if visualization:
            plt.show()


def auto_choose_num_features_to_show(lime_explainer, label, percentage_to_show):
    segments = lime_explainer.segments
    lime_weights = lime_explainer.local_exp[label]
    num_pixels_threshold_in_a_sp = segments.shape[0] * segments.shape[1] // len(np.unique(segments)) // 8

    # l1 norm with filtered weights.
    used_weights = [(tuple_w[0], tuple_w[1]) for i, tuple_w in enumerate(lime_weights) if tuple_w[1] > 0]
    norm = np.sum([tuple_w[1] for i, tuple_w in enumerate(used_weights)])
    normalized_weights = [(tuple_w[0], tuple_w[1] / norm) for i, tuple_w in enumerate(lime_weights)]

    a = 0.0
    n = 0
    for i, tuple_w in enumerate(normalized_weights):
        if tuple_w[1] < 0:
            continue
        if len(np.where(segments == tuple_w[0])[0]) < num_pixels_threshold_in_a_sp:
            continue

        a += tuple_w[1]
        if a > percentage_to_show:
            n = i + 1
            break

    if percentage_to_show <= 0.0:
        return 5

    if n == 0:
        return auto_choose_num_features_to_show(lime_explainer, label, percentage_to_show-0.1)

    return n


def get_cam(image_show, feature_maps, fc_weights, label_index, cam_min=None, cam_max=None):
    _, nc, h, w = feature_maps.shape

    cam = feature_maps * fc_weights[:, label_index].reshape(1, nc, 1, 1)
    cam = cam.sum((0, 1))

    if cam_min is None:
        cam_min = np.min(cam)
    if cam_max is None:
        cam_max = np.max(cam)

    cam = cam - cam_min
    cam = cam / cam_max
    cam = np.uint8(255 * cam)
    cam_img = cv2.resize(cam, image_show.shape[0:2], interpolation=cv2.INTER_LINEAR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_img), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap)
    cam = heatmap + np.float32(image_show)
    cam = cam / np.max(cam)

    return cam


def save_fig(data_, save_outdir, algorithm_name, num_samples=3000):
    import matplotlib.pyplot as plt
    if isinstance(data_, str):
        if algorithm_name == 'cam':
            f_out = f"{algorithm_name}_{data_.split('/')[-1]}.png"
        else:
            f_out = f"{algorithm_name}_{data_.split('/')[-1]}_s{num_samples}.png"
        plt.savefig(
            os.path.join(save_outdir, f_out)
        )
    else:
        n = 0
        if algorithm_name == 'cam':
            f_out = f'cam-{n}.png'
        else:
            f_out = f'{algorithm_name}_s{num_samples}-{n}.png'
        while os.path.exists(
                os.path.join(save_outdir, f_out)
        ):
            n += 1
            if algorithm_name == 'cam':
                f_out = f'cam-{n}.png'
            else:
                f_out = f'{algorithm_name}_s{num_samples}-{n}.png'
            continue
        plt.savefig(
            os.path.join(
                save_outdir, f_out
            )
        )

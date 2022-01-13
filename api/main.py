import logging
import os
import sys

import flask
import pdf2image
import torch
from flask import Flask, render_template, request
from flask import app
from torch.utils.data import DataLoader
import util.misc as utils

import api.config_api
from config import Args
from src.main import get_model, get_transform
from table_datasets import OOSDataset


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Detection(flask.Flask):

    def __init__(self):
        super().__init__(__name__)
        self.config_api = api.config_api.ConfigApi()
        self.args = Args
        self.args.num_workers = 0
        self.args.device = 'cpu'

        self.args.model_load_path = self.config_api.path_table_detector_model
        self.args.type_model = 'detection'

        self.logger = logging.getLogger('Detection API')
        self.logger.setLevel(logging.INFO)

        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        setattr(self.args, 'data_type', 'detection')

        self.logger.info(f'Loading detection model from {self.config_api.path_table_detector_model}')
        self.model_detector, _, self.postprocessors = get_model(self.args, 'cpu')
        self.logger.info(f'Model loaded')
        self.model_structure = None

        @self.route('/predict_table', methods=['GET','POST'])
        def predict_table():
            '''Returns bboxes for tables as x1, y1, x2, y2 and image size'''

            try:
                # get input JSON from back-end
                content = flask.request.get_json()
                # process the input - run the parser (miie package) lib
                pages_converted = pdf2image.convert_from_path(content['input_path'], size=(2000, 3000))

                for i, page in enumerate(pages_converted):
                    page.save(os.path.join(
                        self.config_api.path_temp_image_folder,
                        content['input_path'].split('/')[-1].split('.')[0] + '_' + str(i) + '.png')
                    )

                dataset_test = OOSDataset(
                    img_folder=self.config_api.path_temp_image_folder,
                    transformation=get_transform('detection', "val")
                )

                data_loader_test = DataLoader(dataset_test,
                                              2,
                                              drop_last=False,
                                              collate_fn=utils.collate_fn,
                                              num_workers=0)

                results_all_images = []
                imgs_size = (200,200)
                for samples, targets in data_loader_test:
                    samples = samples.to(self.args.device)
                    targets = [{k: v.to(self.args.device) for k, v in t.items()} for t in targets]

                    outputs = self.model_detector(samples)
                    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                    results = self.postprocessors['bbox'](outputs, orig_target_sizes)
                    results_all_images.extend(results)

                return results_all_images, imgs_size

            except Exception as e:
                for file in os.listdir(self.config_api.path_temp_image_folder):
                    os.remove(os.path.join(self.config_api.path_temp_image_folder, file))

                self.logger.error(e, exc_info=True)

        @self.route('/predict_structure/',methods=['GET','POST'])
        def predict_structure():
            # TODO
            return 'Tvoje mama'

    def reload(self):
        pass



if __name__ == '__main__':
    Detection().run(debug = True)

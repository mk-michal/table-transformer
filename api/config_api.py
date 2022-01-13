import os

from config import Args


class ConfigApi:
    def __init__(self):
        self.path_table_detector_model = os.getenv(
            "PATH_TABLE_DETECTOR_MODEL", "/home/michal/datasets/dataset_1_mil/model_detection.pth"
        )
        self.path_structure_detector_model = os.getenv(
            "PATH_STRUCTURE_DETECTOR_MODEL", "/home/michal/datasets/dataset_1_mil/model_detection.pth"
        )
        self.path_temp_image_folder = os.getenv(
            "PATH_TEMP_IMAGE_FOLDER", "/home/michal/datasets/HGS/visualization/random_output"
        )

        self.path_submissions = os.getenv('PATH_SUBMISSIONS', '/home/michal/datasets/')


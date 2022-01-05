import argparse
import os

import torch
from torch.utils.data import DataLoader

from config import Args
from main import get_transform, get_class_map, get_model
from table_datasets import PDFTablesDataset

import util.misc as utils
from visualize_random_samples import plot_results


def main(args):
    class_map = get_class_map('detection')

    dataset_test = PDFTablesDataset(
        os.path.join(args.data_root_dir,"test"),
        get_transform('detection', "val"),
        do_crop=False,
        make_coco=True,
        image_extension=".jpg",
        xml_fileset="test_filelist_head.txt",
        class_map=class_map
    )
    model, criterion, postprocessors = get_model(args, 'cpu')

    data_loader_test = DataLoader(dataset_test,
                                  2,
                                  drop_last=False,
                                  collate_fn=utils.collate_fn,
                                  num_workers=0)


    i = 0
    for samples, targets in data_loader_test:
        samples = samples.to(args.device)
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        results_subset = []
        outputs_pred_logits = []
        outputs_pred_boxes = []
        for r, logits, boxes in zip(results, outputs['pred_logits'], outputs['pred_boxes']):
            high_scores = r['scores'] > 0.5
            results_subset.append({
                'scores': r['scores'][high_scores],
                'labels': r['labels'][high_scores],
                'boxes': r['boxes'][high_scores]
            })
            outputs_pred_logits.append(logits[high_scores])
            outputs_pred_boxes.append(boxes[high_scores])

        outputs_subset = {'pred_logits': torch.cat(outputs_pred_logits), 'pred_boxes': torch.cat(outputs_pred_boxes)}
        plot_results(samples, targets, outputs, results, dataset_test, args.output_path, plot_gt=False)
        i += len(targets)
        if i >= args.max_samples:
            break




if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',default='/home/michal/datasets/dataset_1_mil/pubtables1m/PubTables-1M-Image_Page_Detection_PASCAL_VOC/PubTables1M-Detection-PASCAL-VOC')
    parser.add_argument('--model_load_path', default='/home/michal/datasets/dataset_1_mil/model_detection.pth')
    parser.add_argument('--output_path', default='/home/michal/datasets/dataset_1_mil/pubtables1m/PubTables-1M-Image_Page_Detection_PASCAL_VOC/visualization')
    parser.add_argument('--max_samples', default=50)
    cfg = parser.parse_args()

    args = Args
    args.device = 'cpu'
    for key, val in cfg.__dict__.items():
        setattr(args, key, val)

    main(args)

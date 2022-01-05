import argparse
import os

import numpy as np
import torch
from matplotlib import patches
from torch.utils.data import DataLoader

import util.misc as utils
from config import Args

from main import get_transform, get_class_map, get_model, get_args
from table_datasets import PDFTablesDataset, read_pascal_voc, OOSDataset
from util.box_ops import box_cxcywh_to_xyxy_np
import matplotlib.pyplot as plt

plt.rcParams['hatch.linewidth'] = 3
plt.rcParams['hatch.linewidth'] = 3

def plt_rectangle(plt, label, x1, y1, x2, y2, class_info):
    '''
    == Input ==

    plt   : matplotlib.pyplot object
    label : string containing the object class name
    x1    : top left corner x coordinate
    y1    : top left corner y coordinate
    x2    : bottom right corner x coordinate
    y2    : bottom right corner y coordinate
    '''
    ax = plt.gca()
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, linewidth=5, edgecolor=class_info[label]['color'], facecolor='none',
        hatch=class_info[label]['hatch'])
    ax.add_patch(rect)


    # plt.plot([x1, x1], [y1, y2], linewidth=linewidth, color=color)
    # plt.plot([x2, x2], [y1, y2], linewidth=linewidth, color=color)
    # plt.plot([x1, x2], [y1, y1], linewidth=linewidth, color=color)
    # plt.plot([x1, x2], [y2, y2], linewidth=linewidth, color=color)


def save_random_img(xml_path: str, output_path: str, data_type: str = 'structure'):
    class_map = get_class_map(data_type)
    bboxes, labels = read_pascal_voc(xml_path, class_map=class_map)
    source_path = '/'.join(xml_path.split('/')[:-2])
    table_path = xml_path.split('/')[-1].split('.')[0] + '.jpg'

    image_path = os.path.join(source_path, 'images', table_path)
    from PIL import Image
    img = Image.open(image_path)

    sorted_classes = np.argsort(labels)
    new_bboxes = [bboxes[i] for i in sorted_classes]
    new_classes = [labels[i] for i in sorted_classes]

    plot_img_with_bboxes(np.array(img), new_bboxes, new_classes, output_path, class_labels=list(class_map.keys()))



def plot_img_with_bboxes(image, bboxes, classes, output_path, class_labels, xyxy = True):
    # read in image
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    hatches = [None, None, None, '/', '/', '\\', None]
    class_labels_info = {cls: {'number': i, 'hatch': h, 'color': c} for i, (cls, h, c) in enumerate(zip(class_labels, hatches, colors))}
    plt.clf()
    plt.figure(figsize=(20, 20))

    index_sorted = np.argsort(classes)
    bboxes = [bboxes[i] for i in index_sorted]
    classes = [classes[i] for i in index_sorted]

    if image.shape[0] == 3:
        image = image.transpose(1,2,0)
    plt.imshow(image)  # plot image

    new_bboxes = []
    if not xyxy:
        bboxes = list(map(box_cxcywh_to_xyxy_np, bboxes))
    for bbox in bboxes:
        new_bboxes.append(
            [bbox[0] * image.shape[1],
             bbox[1] * image.shape[0],
             bbox[2] * image.shape[1],
             bbox[3] * image.shape[0],
             ]
        )

    for bbox, cls in zip(new_bboxes, classes):
        plt_rectangle(plt,
                      label=class_labels[cls],
                      x1=bbox[0],
                      y1=bbox[1],
                      x2=bbox[2],
                      y2=bbox[3],
                      class_info=class_labels_info,
                      )


    text_coord = image.shape[0]
    for i, (cls, col) in enumerate(zip(class_labels, colors)):
        if i in classes:
            text_coord += 30
            plt.text(20, text_coord, cls, color=class_labels_info[cls]['color'], fontsize=20)

    plt.savefig(output_path, dpi=300)  ## show the plot


def main(cfg):

    # get dataset
    args = Args
    args.device = 'cpu'
    args = Args
    args.device = 'cpu'
    for key, val in cfg.__dict__.items():
        setattr(args, key, val)

    class_map = get_class_map(args.data_type)
    model, criterion, postprocessors = get_model(args, args.device)

    if args.custom:
        dataset_test = OOSDataset(args.input_path,get_transform(args.data_type, "val"))
    else:
        dataset_test = PDFTablesDataset(args.input_path,
                                        get_transform(args.data_type, "val"),
                                        do_crop=False,
                                        make_coco=True,
                                        image_extension=".jpg",
                                        xml_fileset="test_filelist.txt",
                                        class_map=class_map,
                                        n_of_samples=args.max_samples)

    data_loader_test = DataLoader(dataset_test,
                                  2,
                                  drop_last=False,
                                  collate_fn=utils.collate_fn,
                                  num_workers=0)


    for samples, targets in data_loader_test:
        samples = samples.to(args.device)
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        plot_results(samples, targets, outputs, results, dataset_test, args.output_path, plot_gt=False)

@torch.no_grad()
def plot_results(samples, targets, outputs, results, dataset, output_path, threshold = 0.5, plot_gt: bool = True):
    for mask, sample, target, result, output in zip(samples.mask, samples.tensors, targets, results, outputs['pred_boxes']):
        s = np.array(sample)
        m = np.array(mask)
        s = s[:, ~m.T[0], :]
        s = s[:, :, ~m[0]]

        img_name = dataset.page_ids[target['image_id'].item()]
        if plot_gt:
            plot_img_with_bboxes(
                s,
                bboxes=np.array(target['boxes']),
                classes=np.array(target['labels']),
                class_labels=['table', 'table column', 'table row', 'table column header',
                              'table projected row header', 'table spanning cell', 'no object'],
                output_path=os.path.join(output_path, f'{img_name}_gt'),
                xyxy=False,
            )
        threshold = 0.6

        bboxes_filtered = np.array(output[result['scores'] > threshold])
        labels_filtered = np.array(result['labels'][result['scores'] > threshold])

        assert len(bboxes_filtered) == len(labels_filtered)

        plot_img_with_bboxes(
            s,
            bboxes=bboxes_filtered,
            classes=labels_filtered,
            class_labels=['table', 'table column', 'table row', 'table column header',
                          'table projected row header', 'table spanning cell', 'no object'],
            output_path=os.path.join(output_path, f'{img_name.split(".")[0]}_pred.png'),
            xyxy=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default='/home/michal/datasets/dataset_1_mil/pubtables1m/visualization/')
    parser.add_argument('--input_path', default='/home/michal/datasets/dataset_1_mil/pubtables1m/PubTables-1M-Image_Table_Structure_PASCAL_VOC/PubTables1M-Structure-PASCAL-VOC/test')
    parser.add_argument('--model_load_path', default='/home/michal/datasets/dataset_1_mil/model.pth')
    parser.add_argument('--max_samples', default=50)
    parser.add_argument('--data_type', default='structure')
    parser.add_argument('--custom', action='store_true', default=False)

    cfg = parser.parse_args()
    os.makedirs(cfg.output_path, exist_ok=True)
    main(cfg)
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

    plt.savefig(output_path, dpi=200)  ## show the plot


def main(cfg, full_pipeline: bool = False):

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
        dataset_test = OOSDataset(img_folder= args.input_path, transformation=get_transform(args.data_type, "val"))
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

    if full_pipeline:
        args.data_type = 'structure'
        args.model_load_path = '/home/michal/datasets/dataset_1_mil/model_structure.pth'
        model_structure, _, postprocessors_structure = get_model(args, args.device)
        get_transform(args.data_type, "val")

    for samples, targets in data_loader_test:
        samples = samples.to(args.device)
        targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        plot_results(samples, targets, outputs, results, dataset_test, args.output_path, plot_gt=False)

        if full_pipeline:
            bboxes = []
            page_ids = []
            for output_bboxes, result_score, target, target_size in zip(outputs['pred_boxes'], results, targets, orig_target_sizes):
                bboxes_filtered = list(output_bboxes[result_score['scores'] > 0.5])
                for bbox in bboxes_filtered:
                    bbox_xyxy = box_cxcywh_to_xyxy_np(bbox.detach())
                    bbox_xyxy = [
                        bbox_xyxy[0] * samples.tensors[0].shape[2] - 30,
                        bbox_xyxy[1] * samples.tensors[0].shape[1] - 30,
                        bbox_xyxy[2] * samples.tensors[0].shape[2] + 30,
                        bbox_xyxy[3] * samples.tensors[0].shape[1] + 30]


                    bbox_resized = resize_bbox(list(map(int, bbox_xyxy)),
                        in_size=(samples.tensors[0].shape[2], samples.tensors[0].shape[1]),
                        out_size=(target_size.tolist()[0], target_size.tolist()[1])
                    )
                    bboxes.append(bbox_resized)
                    page_ids.append(dataset_test.page_ids[target['image_id'].item()])


            # imgs_tables = create_input_data_structure(results_detection=results, samples=samples)
            dataset_structure = OOSDataset(
                img_folder=args.input_path,
                transformation=get_transform('structure', "val"),
                page_ids=page_ids,
                bboxes = bboxes
            )
            data_loader_structure = DataLoader(dataset_structure,
                                          2,
                                          drop_last=False,
                                          collate_fn=utils.collate_fn,
                                          num_workers=0)
            for samples_structure, targets_structure in data_loader_structure:
                outputs_structure = model_structure(samples_structure)
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets_structure], dim=0)
                results_structure = postprocessors_structure['bbox'](outputs_structure, orig_target_sizes)

                plot_results(
                    samples_structure,
                    targets_structure,
                    outputs_structure,
                    results_structure,
                    dataset_structure,
                    args.output_path,
                    plot_gt=False,
                    add_number_to_name=True,
                )




def create_input_data_structure(results_detection, samples, threshold = 0.5):
    # get bboxes
    imgs_tables = []
    page_ids = []
    for result, img, mask, target in zip(results_detection, samples.tensors, samples.mask, targets):
        bboxes_filtered = list(result['boxes'][result['scores'] > threshold])

        for bbox in bboxes_filtered:
            table_img = img[: ]
            # data_bboxes.append(bboxes_filtered)
            img_table = img[:, int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
            imgs_tables.append(img_table)


    return imgs_tables


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    x_scale = float(out_size[0]) / in_size[0]
    y_scale = float(out_size[1]) / in_size[1]
    bbox[0] = x_scale * bbox[0]
    bbox[2] = x_scale * bbox[2]
    bbox[1] = y_scale * bbox[1]
    bbox[3] = y_scale * bbox[3]
    return bbox




@torch.no_grad()
def plot_results(
        samples, targets, outputs, results, dataset, output_path, threshold = 0.5, plot_gt: bool = True,
        add_number_to_name = False, xyxy = False
):
    for i, (mask, sample, target, result, output) in enumerate(
            zip(samples.mask, samples.tensors, targets, results, outputs['pred_boxes'])
    ):
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
        threshold = 0.7

        bboxes_filtered = np.array(output[result['scores'] > threshold])
        labels_filtered = np.array(result['labels'][result['scores'] > threshold])

        assert len(bboxes_filtered) == len(labels_filtered)

        if add_number_to_name:
            img_name = f'{img_name.split(".")[0]}_pred_{i}.png'
        else:
            img_name = f'{img_name.split(".")[0]}_pred.png'

        plot_img_with_bboxes(
            s,
            bboxes=bboxes_filtered,
            classes=labels_filtered,
            class_labels=['table', 'table column', 'table row', 'table column header',
                          'table projected row header', 'table spanning cell', 'no object'],
            output_path=os.path.join(output_path, img_name),
            xyxy=xyxy,
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
    main(cfg, full_pipeline=True)
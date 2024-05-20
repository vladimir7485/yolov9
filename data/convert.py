import os
import cv2
import argparse
import numpy as np
from pylabel import importer
import ultralytics
from ultralytics.data.converter import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold


def convert_coco_mod(
    labels_dir="../coco/annotations/",
    save_dir="coco_converted/",
    use_segments=False,
    use_keypoints=False,
    cls91to80=True,
):
    """
    Converts COCO dataset annotations to a YOLO annotation format  suitable for training YOLO models.

    Args:
        labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
        save_dir (str, optional): Path to directory to save results to.
        use_segments (bool, optional): Whether to include segmentation masks in the output.
        use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
        cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.

    Example:
        ```python
        from ultralytics.data.converter import convert_coco

        convert_coco('../datasets/coco/annotations/', use_segments=True, use_keypoints=False, cls91to80=True)
        ```

    Output:
        Generates output files in the specified output directory.
    """

    # Create dataset directory
    # save_dir = increment_path(save_dir)  # increment if save directory already exists
    save_dir = Path(save_dir)   # if directory already exists then put inside it
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Convert classes
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels"  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            keypoints = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                # cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                cls = coco80[ann["category_id"]] if cls91to80 else ann["category_id"]  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                    if use_segments and ann.get("segmentation") is not None:
                        if len(ann["segmentation"]) == 0:
                            segments.append([])
                            continue
                        elif len(ann["segmentation"]) > 1:
                            s = merge_multi_segment(ann["segmentation"])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        segments.append(s)
                    if use_keypoints and ann.get("keypoints") is not None:
                        keypoints.append(
                            box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        )

            # Write
            cur_file = (fn / f).with_suffix(".txt")
            cur_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cur_file, "a") as file:
                for i in range(len(bboxes)):
                    if use_keypoints:
                        line = (*(keypoints[i]),)  # cls, box, keypoints
                    else:
                        line = (
                            *(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
                        )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

    LOGGER.info(f"COCO data converted successfully.\nResults saved to {save_dir.resolve()}")


def main(path_to_annotations, path_to_images, output_dir):
    print("This is \"prepare_train_val_test.py\" script for TACO dataset.")

    # Conversion with ultralytics API
    if True:
        convert_coco_mod(
            labels_dir=path_to_annotations, 
            save_dir=output_dir,
            use_segments=True,
            cls91to80=True
        )
    
    # Prepare stratified train-val-test splits
    if False:        
        dataset = importer.ImportCoco(os.path.join(path_to_annotations, "annotations.json"), path_to_images=path_to_images, name="TACO_coco")
        df = dataset.df

        # StratifiedGroupKFold (split by 80% - 10% - 10% according to TACO paper)
        y = [int(x) for x in list(df["cat_id"])]        # object category id (label)
        groups = [int(x) for x in list(df["img_id"])]   # image to which object belongs
        X = groups                                      # images, later we'll pick unique samples from it
        samples = np.array(X)
        trains = list()
        vals = list()
        tests = list()
        sgkf4 = StratifiedGroupKFold(n_splits=4, shuffle=True) # usee to pick train
        sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True) # used to pick val/test
        for train, valtest in sgkf4.split(X, y, groups=groups):
            # Got train-valtest split
            train = set(samples[train])
            assert(len(train.intersection(set(samples[valtest]))) == 0) # assert that each image got into single subset

            # Save train split
            train = df.loc[df["img_id"].isin(train)]    # pick image paths for train
            train = list(set(train["img_filename"]))
            trains.append(train)

            # Now split valtest into separate subsets
            _y = (np.array(y)[valtest]).tolist()
            _groups = (np.array(groups)[valtest]).tolist()
            _X = _groups
            _samples = np.array(_X)
            for val, test in sgkf2.split(_X, _y, groups=_groups):
                val =set(_samples[val])
                test = set(_samples[test])
                assert(len(val.intersection(test)) == 0)

                # Save val and test splits
                val = df.loc[df["img_id"].isin(val)]    # pick image paths for test
                test = df.loc[df["img_id"].isin(test)]  # pick image paths for test
                
                val = list(set(val["img_filename"]))
                test = list(set(test["img_filename"]))
                
                vals.append(val)
                tests.append(test)
        
        # Analyze different folds
        # TO-DO
        
        # Save image lists for YOLO
        fold_idx = 0    # which fold to take
        with open(Path(path_to_annotations).parent / "train.txt", 'w') as f:
            f.write('\n'.join([os.path.join(".", "images", x) for x in trains[fold_idx]]))
        with open(Path(path_to_annotations).parent / "val.txt", 'w') as f:
            f.write('\n'.join([os.path.join(".", "images", x) for x in vals[fold_idx]]))
        with open(Path(path_to_annotations).parent / "test.txt", 'w') as f:
            f.write('\n'.join([os.path.join(".", "images", x) for x in tests[fold_idx]]))

    # Print categories to file in darknet format
    if False:
        with open(os.path.join(path_to_annotations, "annotations.json")) as f:
            data = json.load(f)
        with open(Path(path_to_annotations).parent / "taco.txt", 'w') as f:
            f.write('\n'.join(" " + str(item['id']) + ": " + item['name'] for item in data['categories']))

    # Conversion with pylabel API
    if False:
        # Import the dataset into the pylable schema 
        dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="TACO_coco")
        dataset.df.head(5)

        # Analyze annotations
        print(f"Number of images: {dataset.analyze.num_images}")
        print(f"Number of classes: {dataset.analyze.num_classes}")
        print(f"Classes:{dataset.analyze.classes}")
        print(f"Class counts:\n{dataset.analyze.class_counts}")
        print(f"Path to annotations:\n{dataset.path_to_annotations}")

        # Visualize annotations
        # plt.figure(); plt.imshow(dataset.visualize.ShowBoundingBoxes(0))

        # Put images into class subfolders
        if False:
            scale = 512
            class_vis_dir = os.path.join(output_dir, "class_visualization")
            os.makedirs(class_vis_dir, exist_ok=True)
            for idx, item in dataset.df.iterrows():
                print(f"{idx+1} out of {dataset.df.shape[0]} items had been processed...")

                cur_dir = os.path.join(class_vis_dir, item['cat_supercategory'], item['cat_name'])
                os.makedirs(cur_dir, exist_ok=True)

                img_filename = os.path.join(path_to_images, item['img_filename'])
                img = cv2.imread(img_filename)
                h, w = img.shape[:2]
                f = scale / h
                img = cv2.resize(img, dsize=(f * h, w * h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(cur_dir, os.path.basename(img_filename)), img)


        # Export to YOLOv5
        dataset.export.ExportToYoloV5(
            output_path=os.path.join(output_dir, "labels"),
            yaml_file="dataset.yaml"
        )[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-annotations', type=str, required=True, help="Specify path to the coco.json file.")
    parser.add_argument('--path-to-images', type=str, required=False, default="", help="Specify the path to the images (if they are in a different folder than the annotations).")
    parser.add_argument('--name', type=str, required=False, help="Dataset name (optional).")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory.")
    args = parser.parse_args()

    if args.path_to_images == "":
        args.path_to_images = os.path.split(args.path_to_annotations)[0]

    main(args.path_to_annotations, args.path_to_images, args.output_dir)

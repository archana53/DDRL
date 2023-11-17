# Modified from repo - https://github.com/nicehuster/cpm-facial-landmarks
#
import sys
import os, math
import os.path as osp
import numpy as np

from scipy.io import loadmat
from utils import PTSconvert2box, PTSconvert2str
from argparse import ArgumentParser
from pathlib import Path


class AFLWFace:
    # TODO: Face_box presents a concern about the data. Focus on this if things go wrong
    # when there is time
    """A class to hold information about the faces in AFLW.
    Index: index in the order of loading the images
    Image_Path: path to the image for dataloader
    Face_Box: Bounding box around the face. Previous users of AFLW used
              two kinds of boxes - one that AFLW provided and one calculated from
              keypoints. I believe they went through the data and found the bounding boxes
              lacking so they algorithmically created their own. We may need to experiment with
              both kinds of data. The main use of this seems to be for filtering for images that are
              front facing. My main concern is that for the GTL case where box is derived from points,
              by following the logic, wouldn't it always be front-facing?
    Masks: Unsure of the significance as of now. Initial thought was that these are the keypoints that are
           hidden and estimated, but the logic in check_front would be moot then.
    Landmarks: The final set of keypoints
    """
    NUMBER_OF_KEYPOINTS = 19

    def __init__(self, index, name, mask, landmark, box):
        self.image_path = name
        self.face_id = index
        self.face_box = [float(box[0]), float(box[2]), float(box[1]), float(box[3])]
        mask = np.expand_dims(mask, axis=1)
        landmark = landmark.copy()
        self.landmarks = np.concatenate((landmark, mask), axis=1)

    def get_face_size(self, use_box):
        """Based upon whether to use the ALFW bounding box or the algorithmically created one from landmarks,
        returns bounding box and face size
        """
        box = []
        if use_box == "GTL":
            box = PTSconvert2box(self.landmarks.copy().T)
        elif use_box == "GTB":
            box = [
                self.face_box[0],
                self.face_box[1],
                self.face_box[2],
                self.face_box[3],
            ]
        else:
            assert False, "The box indicator not find : {}".format(use_box)
        assert box[2] > box[0], "The size of box is not right [{}] : {}".format(
            self.face_id, box
        )
        assert box[3] > box[1], "The size of box is not right [{}] : {}".format(
            self.face_id, box
        )
        face_size = math.sqrt(float(box[3] - box[1]) * float(box[2] - box[0]))
        box_str = "{:.2f} {:.2f} {:.2f} {:.2f}".format(box[0], box[1], box[2], box[3])
        return box_str, face_size

    def check_front(self):
        """If all the landmarks are within the bounding box, face is facing front"""
        oks = 0
        box = self.face_box
        for idx in range(self.landmarks.shape[0]):
            if bool(self.landmarks[idx, 2]):
                x, y = self.landmarks[idx, 0], self.landmarks[idx, 1]
                if x > self.face_box[0] and x < self.face_box[2]:
                    if y > self.face_box[1] and y < self.face_box[3]:
                        oks = oks + 1
        return oks == self.NUMBER_OF_KEYPOINTS

    def __repr__(self):
        return "{name}(path={image_path}, face-id={face_id})".format(
            name=self.__class__.__name__, **self.__dict__
        )


def filter_images_with_single_face(faces):
    image_paths = {}
    for face in faces:
        path = face.image_path
        if path in image_paths.keys():
            image_paths[path] = image_paths[path] + 1
        else:
            image_paths[path] = 1

    images_to_use = set()

    for path, num in image_paths.items():
        if num == 1:
            images_to_use.add(path)

    save_faces = []
    for face in faces:
        if face.image_path in images_to_use:
            save_faces.append(face)
    return save_faces


def filter_images_with_front_face(allfaces, use_front):
    save_faces = []
    for face in allfaces:
        if use_front == False or face.check_front():
            save_faces.append(face)
    return save_faces


def save_to_list_file(
    root_dir,
    allfaces,
    lst_file,
    image_style_dir,
    annotation_dir,
    use_front,
    use_box,
    use_single_face_images=True,
):
    # Filtering whether only front faces are to be used
    save_faces = []

    if use_single_face_images is True:
        save_faces = filter_images_with_single_face(allfaces)

    save_faces = filter_images_with_front_face(save_faces, use_front)

    print("Prepare to save {} face images into {}".format(len(save_faces), lst_file))

    with open(lst_file, "w") as lst_file:
        all_face_sizes = []
        index = 0
        for face in save_faces:
            image_path = face.image_path
            sub_dir, base_name = image_path.split("/")
            annot_dir = root_dir / annotation_dir / sub_dir
            pts_file_name = base_name.split(".")[0] + "-{}.pts".format(face.face_id)
            annot_path = root_dir / annot_dir / pts_file_name
            annot_dir.mkdir(exist_ok=True, parents=True)
            image_path = root_dir / image_style_dir / image_path
            assert image_path.is_file(), "The image [{}/{}] {} does not exist".format(
                index, len(save_faces), image_path
            )

            # pts_file corresponds to the <keypoint_annotation_path> used by the dataloader
            if not annot_path.is_file():
                pts_str = PTSconvert2str(face.landmarks.T)
                with open(annot_path, "w") as pts_file:
                    pts_file.write("{}".format(pts_str))
            else:
                pts_str = None

            box_str, face_size = face.get_face_size(use_box)

            lst_file.write(
                "{} {} {} {}\n".format(image_path, annot_path, box_str, face_size)
            )
            all_face_sizes.append(face_size)
            index = index + 1

    all_faces = np.array(all_face_sizes)
    print("all faces : mean={}, std={}".format(all_faces.mean(), all_faces.std()))


# Right now only separate demarcations for all and front facing is done.
# In the next step will process face crops

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Aflw data processor. Expected directory structure: "
        "modules/data/aflw_dataset/images/0/*.jpg, "
        "modules/data/aflw_dataset/images/2/*.jpg, "
        "modules/data/aflw_dataset/images/3/*.jpg, "
        "modules/data/aflw_dataset/ALFWinfo_release.mat"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/suyash/Gatech/DDRL/modules/data/aflw_dataset",
        help="Path to extracted and reorganized aflw dataset",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    SAVE_DIR = root_dir / "AFLW_lists"
    SAVE_DIR.mkdir(exist_ok=True)
    image_dir = "images"
    annot_dir = osp.join("processed", "annotations")
    print("AFLW image dir : {}".format(image_dir))
    print("AFLW annotation dir : {}".format(annot_dir))

    mat_path = root_dir / "AFLWinfo_release.mat"
    aflwinfo = dict()
    mat = loadmat(str(mat_path))
    total_image = 24386

    aflwinfo["name-list"] = []

    # load name-list
    for i in range(total_image):
        name = mat["nameList"][i, 0][0]
        aflwinfo["name-list"].append(name)

    aflwinfo["mask"] = mat["mask_new"].copy()

    # Set individual values into 19 coordinates
    aflwinfo["landmark"] = mat["data"].reshape((total_image, 2, 19))
    aflwinfo["landmark"] = np.transpose(aflwinfo["landmark"], (0, 2, 1))

    aflwinfo["box"] = mat["bbox"].copy()
    allfaces = []
    for i in range(total_image):
        face = AFLWFace(
            i,
            aflwinfo["name-list"][i],
            aflwinfo["mask"][i],
            aflwinfo["landmark"][i],
            aflwinfo["box"][i],
        )
        allfaces.append(face)

    USE_BOXES = ["GTL", "GTB"]
    for USE_BOX in USE_BOXES:
        save_to_list_file(
            root_dir,
            allfaces,
            osp.join(SAVE_DIR, "all.{}".format(USE_BOX)),
            image_dir,
            annot_dir,
            False,
            USE_BOX,
        )
        save_to_list_file(
            root_dir,
            allfaces,
            osp.join(SAVE_DIR, "front.{}".format(USE_BOX)),
            image_dir,
            annot_dir,
            True,
            USE_BOX,
        )

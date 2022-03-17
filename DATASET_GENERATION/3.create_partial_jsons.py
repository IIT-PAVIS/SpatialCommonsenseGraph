import numpy as np
import scipy
from pathlib import Path
from plyfile import PlyData
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

# import mpi4py
from tqdm import tqdm
import json

import scipy.spatial.distance as dist
import sys
import ray


ITEMS_TO_SKIP = [1, 2, 22, 31]  # FLOOR, WALL, CEILING, PERSON


def get_transf_and_bb(ref_pcd):

    ref_pcd = copy.deepcopy(ref_pcd)

    obj_flat = copy.deepcopy(ref_pcd)
    obj_flat[:, 2] = 0

    pca = PCA(3)
    pca.fit(obj_flat)
    components = pca.components_

    transf = np.array(
        [components[0], [-components[0, 1], components[0, 0], 0], [0, 0, 1]]
    )
    mean = pca.mean_

    rotated_ref_pcd = (ref_pcd) @ transf.T
    width = rotated_ref_pcd[:, 0].max() - rotated_ref_pcd[:, 0].min()
    length = rotated_ref_pcd[:, 1].max() - rotated_ref_pcd[:, 1].min()
    height = rotated_ref_pcd[:, 2].max() - rotated_ref_pcd[:, 2].min()

    return mean, transf, (width, length, height)


def transform_pcd(mean, transf, pcd):
    pcd = copy.deepcopy(pcd)

    pcd = pcd - mean

    pcd = np.matmul(pcd, transf.T)

    return pcd, transf


def transform_pcd_old(ref_pcd, pcd):
    ref_pcd = copy.deepcopy(ref_pcd)
    pcd = copy.deepcopy(pcd)

    obj_flat = ref_pcd
    obj_flat[:, 2] = 0

    pca = PCA(3)
    pca.fit(obj_flat)
    components = pca.components_

    transf = np.array(
        [components[0], [-components[0, 1], components[0, 0], 0], [0, 0, 1]]
    )
    mean = pca.mean_
    pcd = pcd - mean

    pcd = np.matmul(pcd, transf.T)

    rotated_ref_pcd = (ref_pcd) @ transf.T
    width = rotated_ref_pcd[:, 0].max() - rotated_ref_pcd[:, 0].min()
    length = rotated_ref_pcd[:, 1].max() - rotated_ref_pcd[:, 1].min()
    height = rotated_ref_pcd[:, 2].max() - rotated_ref_pcd[:, 2].min()
    # ll = np.array([rotated_ref_pcd[:, 0].min(), rotated_ref_pcd[:, 1].min()])
    # new=[ll,ll+[width,0],ll+[width,length],ll+[0,length],ll]
    # new=np.array(new)
    return pcd, transf, (width, length, height)


def test_eval_scene(scene_ply_name):

    ply = PlyData.read(scene_ply_name)
    vertex = ply["vertex"]
    xyz = np.array([vertex["x"], vertex["y"], vertex["z"]]).T
    instance = np.array(vertex["instance"])
    label = np.array(vertex["label"])

    label_set = set(np.unique(label)) - {0}

    for lab in label_set:
        label_ind = label == lab
        if lab < 4:
            continue

        ind_set = np.unique(instance[label_ind])
        for inst in ind_set:

            instance_ind = instance == inst

            obj = xyz[label_ind & instance_ind]
            obj_flat = obj.copy()
            obj_flat[:, 2] = 0
            pca = PCA(3)
            pca.fit(obj_flat)
            components = pca.components_

            axis0 = components[0]  # + pca.mean_
            axis0 = np.array([axis0 * -1 + pca.mean_, axis0 * 1 + pca.mean_])

            axis1 = components[1]  # + pca.mean_
            axis1 = np.array([axis1 * -1 + pca.mean_, axis1 * 1 + pca.mean_])

            axis2 = components[2]  # + pca.mean_
            axis2 = np.array([axis2 * -1 + pca.mean_, axis2 * 1 + pca.mean_])

            print(lab)
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2])
            ax.plot(axis0[:, 0], axis0[:, 1], axis0[:, 2], "r")
            ax.plot(axis1[:, 0], axis1[:, 1], axis1[:, 2], "g")
            ax.plot(axis2[:, 0], axis2[:, 1], axis2[:, 2], "y")

            transformer_obj = transform_pcd(obj, obj)
            ax.scatter(
                transformer_obj[:, 0],
                transformer_obj[:, 1],
                transformer_obj[:, 2],
                color="orange",
            )
            plt.show()
            pass
    pass


@ray.remote
def calc_scene_distances(list_):  # (partial_scene, complete_scene):
    # print("start")
    partial_scene = list_[0]
    complete_scene = list_[1]
    out_dir = Path("partial_jsons")
    out_dir.mkdir(exist_ok=True)

    # print("ready")
    scene_pcd = PlyData.read(str(partial_scene))
    vert = scene_pcd["vertex"]
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], 1)
    label = np.array(vert["label"])
    instance = np.array(vert["instance"])

    j = {}
    j["center"] = []
    j["labels"] = []
    j["distances"] = []
    j["relative_pos"] = []
    j["masked"] = []
    j["bb_shape"] = []

    seen_instances = []

    ##############    objects in partial

    for obj_i in range(1, 38):
        if obj_i in ITEMS_TO_SKIP:
            continue
        ind_obj_i = label == obj_i
        instances_i = sorted(np.unique(instance[ind_obj_i]))
        for inst_i in instances_i:

            pcd_i = xyz[(instance == inst_i) & ind_obj_i]
            if pcd_i.shape[0] < 50:
                continue
            mean, transf, bb = get_transf_and_bb(pcd_i)

            seen_instances.append(inst_i)
            kdtree = scipy.spatial.cKDTree(pcd_i)
            j["labels"].append(obj_i)
            j["center"].append(((pcd_i.max(0) + pcd_i.min(0)) / 2).tolist())
            j["masked"].append(0)
            j["bb_shape"].append([bb[0], bb[1], bb[2]])
            tmp_dist = []
            tmp_rel = []
            for obj_j in range(1, 38):
                if obj_j in ITEMS_TO_SKIP:
                    continue

                ind_obj_j = label == obj_j
                instances_j = sorted(np.unique(instance[ind_obj_j]))
                for inst_j in instances_j:
                    pcd_j = xyz[(instance == inst_j) & ind_obj_j]
                    if pcd_j.shape[0] < 50:
                        continue

                    cur_min = np.inf
                    # for i in range(0, pcd_j.shape[0], 1300):
                    # cur_min = min(
                    #    dist.cdist(pcd_i, pcd_j[i : i + 300]).min(), cur_min
                    # )
                    cur_min = kdtree.query(pcd_j)[0].min()
                    # pcd_j_new, _ = transform_pcd_old(pcd_i, pcd_j)
                    pcd_j_new, _ = transform_pcd(mean, transf, pcd_j)

                    rel = ((pcd_j_new.max(0) + pcd_j_new.min(0)) / 2).tolist()
                    tmp_dist.append(cur_min)
                    tmp_rel.append(rel)
            j["distances"].append(tmp_dist)

            j["relative_pos"].append(tmp_rel)

        ########### complete scene for missing items

    complete_pcd = PlyData.read(str(complete_scene))
    complete_vert = complete_pcd["vertex"]
    complete_xyz = np.stack(
        [complete_vert["x"], complete_vert["y"], complete_vert["z"]], 1
    )
    complete_label = np.array(complete_vert["label"])
    complete_instance = np.array(complete_vert["instance"])

    instances_j_seen = []
    j["masked_center"] = []
    j["masked_label"] = []
    j["distance_to_masked"] = []
    j["relative_pos_to_masked"] = []

    for obj_i in range(1, 38):
        if obj_i in ITEMS_TO_SKIP:
            continue

        ind_obj_i = label == obj_i
        instances_i = sorted(np.unique(instance[ind_obj_i]))
        for inst_i in instances_i:
            pcd_i = xyz[(instance == inst_i) & ind_obj_i]
            if pcd_i.shape[0] < 50:
                continue

            mean, transf, bb = get_transf_and_bb(pcd_i)
            kdtree = scipy.spatial.cKDTree(pcd_i)
            tmp_dist = []
            tmp_rel = []
            for obj_j in range(1, 38):
                if obj_j in ITEMS_TO_SKIP:
                    continue
                ind_obj_j = complete_label == obj_j
                instances_j = sorted(np.unique(complete_instance[ind_obj_j]))
                for inst_j in instances_j:
                    if inst_j in seen_instances:
                        continue

                    pcd_j = complete_xyz[(complete_instance == inst_j) & ind_obj_j]
                    if inst_j not in instances_j_seen:
                        j["masked_center"].append(
                            ((pcd_j.max(0) + pcd_j.min(0)) / 2).tolist()
                        )
                        j["masked_label"].append(obj_j)
                        instances_j_seen.append(inst_j)

                    cur_min = np.inf
                    # for i in range(0, pcd_j.shape[0], 1300):
                    # cur_min = min(
                    #    dist.cdist(pcd_i, pcd_j[i : i + 300]).min(), cur_min
                    # )
                    cur_min = kdtree.query(pcd_j)[0].min()
                    # pcd_j_new, _ = transform_pcd_old(pcd_i, pcd_j)
                    pcd_j_new, _ = transform_pcd(mean, transf, pcd_j)

                    rel = ((pcd_j_new.max(0) + pcd_j_new.min(0)) / 2).tolist()
                    tmp_dist.append(cur_min)
                    tmp_rel.append(rel)
            j["distance_to_masked"].append(tmp_dist)

            j["relative_pos_to_masked"].append(tmp_rel)

    if len(j['masked'])==0 or len(j['labels'])<3:
        return

    with (out_dir / f"{complete_scene.stem}_{partial_scene.stem}.json").open("w") as f:
        json.dump(j, f)

    return j


def main():
    scene_ply_name = sorted(list(Path("annotated_ply").glob("*.ply")))

    ROOT_PARTIAL_DIR = Path("partial_pcds")
    SCENE_DIR = sorted([x for x in ROOT_PARTIAL_DIR.iterdir() if x.is_dir()])
    ROOT_COMPLETE_DIR = Path("annotated_ply")
    jobs = []
    arg_list = []
    for scene in SCENE_DIR:
        partial_scenes = sorted(list(scene.glob("*.ply")))
        scene_name = scene.stem

        complete = ROOT_COMPLETE_DIR / f"{scene_name}.ply"

        for partial in partial_scenes:
            jobs.append(calc_scene_distances.remote((partial, complete)))

            pass

    pass

    with tqdm(total=len(jobs)) as pbar:

        unfinished = jobs
        num_ret = min(len(unfinished), 3)
        while unfinished:

            num_ret = min(len(unfinished), 3)
            ## Returns the first ObjectRef that is ready.
            # print(len(unfinished))
            # print(num_ret)
            finished, unfinished = ray.wait(unfinished, num_returns=num_ret)
            result = ray.get(finished)
            pbar.update(num_ret)
            sys.stdout.flush()


if __name__ == "__main__":
    ray.init()
    main()

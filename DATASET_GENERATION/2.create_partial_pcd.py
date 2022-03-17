# %%
from collections import defaultdict
import open3d as o3d
import open3d.visualization as o3d_vis
import ray
from pathlib import Path
import numpy as np
import copy
from PIL import Image

import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.stats as stats
import tqdm
import json
import gc
import scipy.spatial.distance as distance
from plyfile import PlyData, PlyElement
import argparse
import sys
import os


@ray.remote  # (num_cpus=6)
def create_pcd_rgb(scene_path, color, depth, new_width, new_height, intrinsic, extr):
    from PIL import Image
    import open3d as o3d
    import numpy as np

    rgb = color  # Image.open(scene_path / "color" / f"{img}.jpg")
    depth = depth  # Image.open(scene_path / "depth" / f"{img}.png")

    new_rgb = np.array(rgb.resize((new_width, new_height), resample=Image.NEAREST))
    new_depth = np.array(depth.resize((new_width, new_height), resample=Image.NEAREST))
    ### RESOLVE ISSUES WITH BLACK HOLES

    # image_result = inpaint.inpaint_biharmonic(new_depth, mask)

    # fig,ax=plt.subplots(1,3)

    # ax[0].imshow(new_depth)
    # ax[1].imshow(mask.astype(np.uint8))
    # ax[2].imshow(image_result)

    # #fig.colorbar()
    # plt.show()

    ### extract

    extrinsic = extr  # np.loadtxt(scene_path / "pose" / f"{img}.txt")
    new_depth[new_depth < 600] = 10000

    # plt.hist(new_depth.reshape(-1))
    # plt.show()
    rgb_o3d = o3d.geometry.Image(new_rgb)
    depth_o3d = o3d.geometry.Image(new_depth.astype(np.uint16))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            new_width,
            new_height,
            intrinsic[0][0],
            intrinsic[1][1],
            intrinsic[0][2],
            intrinsic[1][2],
        ),
    )
    pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pcd.transform(extrinsic)
    # pcd_denoised, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.2)
    return (
        (np.asarray(pcd.points), np.asarray(pcd.colors)),
        # (np.asarray(pcd_denoised.points), np.asarray(pcd_denoised.colors)),
    )


@ray.remote
def create_pcd_sem(scene_path, color, depth, new_width, new_height, intrinsic, extr):
    from PIL import Image
    import open3d as o3d
    import numpy as np

    rgb = color  # Image.open(scene_path / "instance" / f"{img}.png")
    depth = depth  # Image.open(scene_path / "depth" / f"{img}.png")

    new_rgb = np.array(rgb.resize((new_width, new_height), resample=Image.NEAREST))
    new_depth = np.array(depth.resize((new_width, new_height), resample=Image.NEAREST))
    ### RESOLVE ISSUES WITH BLACK HOLES

    extrinsic = extr  # np.loadtxt(scene_path / "pose" / f"{img}.txt")
    new_depth[new_depth < 600] = 10000

    ##### Semantic

    rgb_o3d = o3d.geometry.Image(
        (new_rgb[:, :, np.newaxis] // 1000).repeat(3, 2).astype(np.uint8)
    )
    depth_o3d = o3d.geometry.Image(new_depth.astype(np.uint16))

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False,
    )

    pcd_sem = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            new_width,
            new_height,
            intrinsic[0][0],
            intrinsic[1][1],
            intrinsic[0][2],
            intrinsic[1][2],
        ),
    )

    pcd_sem.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pcd_sem.transform(extrinsic)

    ###
    rgb_o3d = o3d.geometry.Image(
        (new_rgb[:, :, np.newaxis] % 1000).repeat(3, 2).astype(np.uint8)
    )
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, depth_o3d, convert_rgb_to_intensity=False,
    )

    pcd_inst = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            new_width,
            new_height,
            intrinsic[0][0],
            intrinsic[1][1],
            intrinsic[0][2],
            intrinsic[1][2],
        ),
    )

    pcd_inst.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    pcd_inst.transform(extrinsic)

    assert pcd_sem.points == pcd_inst.points
    return (
        (
            np.asarray(pcd_sem.points),
            np.asarray(pcd_sem.colors),
            np.asarray(pcd_inst.colors),
        ),
        # (np.asarray(pcd_denoised.points), np.asarray(pcd_denoised.colors)),
    )


def pcd_from_points(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(deepcopy(points))
    pcd.colors = o3d.utility.Vector3dVector(deepcopy(colors))
    # down=pcd.voxel_down_sample(voxel_size=0.05)
    # down,_=down.remove_statistical_outlier(nb_neighbors=30,
    #                                                    std_ratio=2.0)
    return pcd


@ray.remote  # (num_cpus=6)
def calc_mode(colors, element_list):
    return stats.mode(colors[element_list])[0]


@ray.remote
def calc_min(xA, xB):
    temp = distance.cdist(xA, xB).T.argmin(0)
    return temp


@ray.remote(num_cpus=os.cpu_count() // 2)
def preprocess_scene(scene_path, new_width, new_height, subseq):
    images = (scene_path / "label").glob("*")
    file_names = sorted([x.stem for x in images])[:subseq]

    intr_color = np.loadtxt(scene_path / "intrinsics_color.txt")
    intr_depth = np.loadtxt(scene_path / "intrinsics_depth.txt")

    new_intr_rgb = rescale_intrinsic(intr_color, new_width, new_height)
    new_intr_depth = rescale_intrinsic(intr_depth, new_width, new_height)
    ret = []
    for img in file_names:

        ret.append(
            create_pcd_rgb.remote(
                scene_path,
                Image.open(scene_path / "color" / f"{img}.jpg"),
                Image.open(scene_path / "depth" / f"{img}.png"),
                new_width,
                new_height,
                new_intr_depth,
                np.loadtxt(scene_path / "pose" / f"{img}.txt"),
            )
        )

    # ret_rgb = [
    #     create_pcd_rgb.remote(
    #         scene_path, color, depth, new_width, new_height, new_intr_depth, extr
    #     )
    #     for img in file_names
    # ]
    res = ray.get(ret)

    pcds_n = [pcd_from_points(x[0][0], x[0][1]) for x in res]

    del ret, res
    gc.collect()
    # del ret, res

    pcd = o3d.geometry.PointCloud()
    while len(pcds_n) > 0:
        p = pcds_n.pop(0)
        pcd += p

    pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
    pcd_downscaled_points = np.array(pcd_down.points)

    pcd_original = PlyData.read(str(Path("annotated_ply") / f"{scene_path.stem}.ply"))
    pcd_vertex = pcd_original["vertex"]
    original_points = np.stack(
        [pcd_vertex["x"], pcd_vertex["y"], pcd_vertex["z"]], 1
    ).astype(np.float64)
    original_color = (
        np.stack([pcd_vertex["red"], pcd_vertex["green"], pcd_vertex["blue"]], 1) / 255
    )
    original_label = pcd_vertex["label"]
    original_instance = pcd_vertex["instance"]

    # pcd_original = o3d.io.read_point_cloud(
    #    str(scene_path / f"{scene_path.stem}_vh_clean_2.labels.ply"), format="ply"
    # )
    # pcd_original_points = np.array(pcd_original.points)
    # pcd_original_colors = np.array(pcd_original.colors)

    matching = []
    step_size = 100
    orig = ray.put(original_points)
    rem = []
    for i in range(0, pcd_downscaled_points.shape[0], step_size):
        # print(pcd_downscaled_points.dtype, original_points.dtype)
        # temp = distance.cdist(
        #    pcd_downscaled_points[i : i + step_size], original_points
        # ).T.argmin(0)
        # matching.append(temp)
        rem.append(calc_min.remote(pcd_downscaled_points[i : i + step_size], orig))
    matching = ray.get(rem)
    new_col = [original_color[x] for x in np.concatenate(matching)]
    new_inst = [original_instance[x] for x in np.concatenate(matching)]
    new_label = [original_label[x] for x in np.concatenate(matching)]
    new_points = [original_points[x] for x in np.concatenate(matching)]

    descr_lab = ["x", "y", "z", "red", "green", "blue", "label", "instance"]
    descr_type = ["f", "f", "f", "f", "f", "f", "i4", "i4"]

    types = list(zip(descr_lab, descr_type))

    tmp = np.empty(len(new_col), types)

    tmp["x"] = np.array(new_points)[:, 0]
    tmp["y"] = np.array(new_points)[:, 1]
    tmp["z"] = np.array(new_points)[:, 2]
    tmp["red"] = np.array(new_col)[:, 0]
    tmp["green"] = np.array(new_col)[:, 1]
    tmp["blue"] = np.array(new_col)[:, 2]
    tmp["label"] = np.array(new_label)
    tmp["instance"] = np.array(new_inst)

    new_v = PlyElement.describe(tmp, "vertex")
    new_p = PlyData([new_v], text=False)

    out_dir_ply = Path("partial_pcds") / scene_path.stem
    out_dir_ply.mkdir(exist_ok=True, parents=True)
    new_p.write(str(out_dir_ply / f"{subseq:04d}.ply"))

    # o3d.visualization.draw_geometries([pcd_down])

    # pcd_down.colors = o3d.utility.Vector3dVector(new_col)
    # o3d.visualization.draw_geometries([pcd_down])
    objects = defaultdict(lambda: [])
    for obj_idx in set(new_inst):
        idx_ = np.where(new_inst == obj_idx)[0]
        lab = np.array(new_label)[idx_][0]
        pcd_inst = pcd_down.select_by_index(idx_)
        bb = pcd_inst.get_axis_aligned_bounding_box()
        objects[int(lab)].append(
            {"min_bb": bb.min_bound.tolist(), "max_bb": bb.max_bound.tolist()}
        )

    # out_dir_json = Path("out_json_sing") / scene_path.stem
    # out_dir_json.mkdir(exist_ok=True, parents=True)

    # json.dump(objects, (out_dir_json / f"{subseq:04d}.json").open("w"))

    pass


def rescale_intrinsic(intrinsic, new_w, new_h):

    original_w = 2 * intrinsic[0][2]
    original_h = 2 * intrinsic[1][2]
    ratio_w = new_w / original_w
    ratio_h = new_h / original_h
    new_intr = copy.deepcopy(intrinsic)

    new_intr[0] = new_intr[0] * ratio_w
    new_intr[1] = new_intr[1] * ratio_h

    return new_intr


def main():
    start_index = 0
    scenes = sorted(Path("scannet_frames_25k").glob("*"))
    jobs = []
    count = 0
    with tqdm.tqdm(total=len(scenes)) as pbar:
        for id_, s in enumerate(scenes[start_index:1]):
            list_subseq = list(s.glob("color/*"))
            for subseq in range(1, len(list_subseq) + 1):

                print(
                    f"{id_+start_index:04d} / {len(scenes)}  :  {s.stem}  - {subseq}/{len(list_subseq)+1}"
                )

                jobs.append(preprocess_scene.remote(s, 1280, 960, subseq))

                if len(jobs) >= 3:
                    chunk = jobs[:3]
                    jobs = jobs[3:]
                    results = ray.get(chunk)

            pbar.update()
            sys.stdout.flush()

            # gc.collect()
        # ray.shutdown()
    # final = ray.get(jobs)
    while len(jobs) > 0:
        chunk = jobs[:3]
        jobs = jobs[3:]
        results = ray.get(chunk)


def main2():

    # start_index = 0
    scenes = sorted(Path("scannet_frames_25k").glob("*"))
    jobs = []
    count = 0
    # list_jobs=[]
    args = []
    for id_, s in enumerate(scenes):
        list_subseq = list(s.glob("color/*"))
        for subseq in range(1, len(list_subseq) + 1):

            # print(
            # f"{id_+start_index:04d} / {len(scenes)}  :  {s.stem}  - {subseq}/{len(list_subseq)+1}"
            # )
            args.append((s, 1280, 960, subseq))

            # jobs.append(preprocess_scene.remote(s, 1280, 960, subseq))

    with tqdm.tqdm(total=len(args)) as pbar:
        for arg in args:

            preprocess_scene(*arg)
            pbar.update()
            sys.stdout.flush()
        # for i
        # a = ray.get(l)
        # unfinished = jobs
        # num_ret = min(len(unfinished), 3)
        # while unfinished:
        ## Returns the first ObjectRef that is ready.
        # finished, unfinished = ray.wait(unfinished, num_returns=num_ret)
        # result = ray.get(finished)
        # pbar.update(num_ret)
        # sys.stdout.flush()


def main3():

    scenes = sorted(Path("scannet_frames_25k").glob("*"))
    jobs = []
    count = 0
    for id_, s in enumerate(scenes):
        list_subseq = list(s.glob("color/*"))
        for subseq in range(1, len(list_subseq) + 1):

            # print(
            # f"{id_+start_index:04d} / {len(scenes)}  :  {s.stem}  - {subseq}/{len(list_subseq)+1}"
            # )
            # args.append((s, 1280, 960, subseq))

            jobs.append(preprocess_scene.remote(s, 1280, 960, subseq))

    with tqdm.tqdm(total=len(jobs)) as pbar:
        # for arg in args:

        # preprocess_scene(*arg)
        # pbar.update()
        # sys.stdout.flush()
        # for i
        # a = ray.get(l)
        unfinished = jobs
        num_ret = min(len(unfinished), 3)
        while unfinished:
            ## Returns the first ObjectRef that is ready.
            finished, unfinished = ray.wait(unfinished, num_returns=num_ret)
            # result = ray.get(finished)
            pbar.update(num_ret)
            sys.stdout.flush()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="preprocess scanned scenes")
    # parser.add_argument("--redis_pwd", required=True, type=str, help="redis pwd")

    # parser.add_argument("--head_ip", required=True, type=str, help="ray head ip")
    # args = parser.parse_args()
    # ray.init(address="auto", _redis_password="5241590000000000")
    # ray.init(address="auto", _redis_password="5241590000000000")
    # ray.init(address='auto', _redis_password='5241590000000000')
    # ray.init(address=args.head_ip, _redis_password=args.redis_pwd)
    ray.init()
    main()


# %%

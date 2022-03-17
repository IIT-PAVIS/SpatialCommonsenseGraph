from plyfile import PlyData, PlyElement, PlyProperty
from pathlib import Path
import json
import pandas as pd
import numpy as np

import open3d as o3d
import ray
import tqdm
import sys

DEBUG = False


@ray.remote
def parse_scene(scene_folder: Path, out_folder: Path):

    original_ply = PlyData.read(
        str(scene_folder / f"{scene_folder.stem}_vh_clean_2.labels.ply")
    )

    json_aggregation = json.load(
        (scene_folder / f"{scene_folder.stem}.aggregation.json").open("r")
    )

    json_segments = json.load(
        (scene_folder / f"{scene_folder.stem}_vh_clean_2.0.010000.segs.json").open("r")
    )

    instance_ids_array = np.ones(original_ply["vertex"]["x"].shape[0], dtype=int) * -1

    for instance_info in json_aggregation["segGroups"]:
        instance = instance_info["objectId"]
        # print(instance_info["label"])
        seg = instance_info["segments"]

        for s in seg:
            ind = np.where(np.array(json_segments["segIndices"]) == s)[0]
            instance_ids_array[ind] = instance

            pass
        points_indices = np.where(instance_ids_array == instance)[0]

        if DEBUG:
            points = np.stack(
                [
                    original_ply["vertex"]["x"],
                    original_ply["vertex"]["y"],
                    original_ply["vertex"]["z"],
                ],
                1,
            )
            points_instance = points[points_indices]
            color = np.array([1.0, 0.0, 0.0], dtype=np.float)[np.newaxis, :].repeat(
                points_instance.shape[0], 0
            )

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_instance)
            pcd.colors = o3d.utility.Vector3dVector(color)

            o3d.visualization.draw_geometries([pcd])

    v = original_ply["vertex"]
    tmp = np.empty(len(v.data), v.data.dtype.descr + [("instance", "i4")])
    for name in v.data.dtype.fields:
        tmp[name] = v[name]
    tmp["instance"] = instance_ids_array

    v = PlyElement.describe(tmp, "vertex")
    p = PlyData([v, original_ply["face"]], text=False)

    p.write(str(out_folder / f"{scene_folder.stem}.ply"))

    pass


def main():
    OUT_folder: Path = Path("annotated_ply")
    OUT_folder.mkdir(exist_ok=True)
    root_folder = Path("scans")
    scenes = [x for x in root_folder.glob("*")]
    l = []
    for s in tqdm.tqdm(scenes):
        l.append(parse_scene.remote(s, OUT_folder))

    print(
        "\n\nCreating the annotated PCD for the full scenes in the folder 'annotated_ply', please wait, this may take a while\n"
    )
    with tqdm.tqdm(total=len(scenes)) as pbar:
        # for i
        # a = ray.get(l)
        unfinished = l
        num_ret = min(len(unfinished), 3)
        while unfinished:
            # Returns the first ObjectRef that is ready.
            finished, unfinished = ray.wait(unfinished, num_returns=num_ret)
            result = ray.get(finished)
            pbar.update(num_ret)
            sys.stdout.flush()


if __name__ == "__main__":
    ray.init()
    main()

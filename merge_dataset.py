import os
import shutil
import random

def extract_and_rename_folders(source_dirs, destination_dir):
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    folder_count = 1

    for source_dir in source_dirs:
        # 获取源文件夹中的所有子文件夹
        all_folders = [f for f in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, f))]
        all_folders = sorted(all_folders)
        selected_folders = all_folders[:1024]

        for folder in selected_folders:
            source_folder_path = os.path.join(source_dir, folder)
            new_folder_name = f"{folder_count:06d}"  # 重命名
            dest_folder_path = os.path.join(destination_dir, new_folder_name)

            # 复制文件夹并重命名
            shutil.copytree(source_folder_path, dest_folder_path)
            print(f"Folder '{folder}' copied and renamed to '{new_folder_name}'")

            folder_count += 1  # 递增以保证文件夹名唯一

if __name__ == "__main__":
    # 输入要提取文件夹的目录列表
    source_dirs = [
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_graspTargetObj_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_graspTargetObj_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/3_objs_graspTargetObj_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushFront_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_pushFront_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_pushLeft_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_pushLeft_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_knockOver_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_knockOver_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/1_objs_placeTargetObj_Right_0512',
      '/data2/liangxiwen/zkd/datasets/dataGen/DATA/2_objs_moveNear_Right_0512',
]
    
    # 新建的目标文件夹路径
    destination_dir = '/data2/liangxiwen/zkd/PIVOT-R/datasets/train_data'  # 修改为你的目标文件夹路径

    random.seed(42)
    # 执行提取和重命名操作
    extract_and_rename_folders(source_dirs, destination_dir)

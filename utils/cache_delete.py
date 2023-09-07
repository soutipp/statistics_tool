import os
import time


def delete_old_files(target_folder, max_size=15 * 1024 * 1024 * 1024, min_size=8 * 1024 * 1024 * 1024,
                     max_age=4 * 7 * 24 * 60 * 60 + 2 * 24 * 60 * 60):
    """
    :param target_folder: 目标文件夹路径
    :param max_size: 最大文件大小，单位为字节，默认为15GB
    :param min_size: 最小文件大小，单位为字节，默认为8GB
    :param max_age: 最大文件修改时间，单位为秒，默认为30天
    """
    total_size = 0
    del_files = []
    now = time.time()

    for root, _, files in os.walk(target_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
            total_size += file_size
            del_files.append((file_path, file_size, file_mtime))

    del_files.sort(key=lambda x: x[2])  # 按修改时间排序,默认升序

    if total_size >= max_size or now - del_files[0][2] >= max_age:
        print(f"{target_folder} 中的缓存文件超过 {max_size / 1024 / 1024 / 1024} GB, 开始删除缓存...")
        for file_path, file_size, file_mtime in del_files:
            if total_size <= min_size and now - file_mtime < max_age:
                break
            os.remove(file_path)
            print(f"{file_path} ({file_size / 1024 / 1024 / 1024} GB) 已被删除.")
            total_size -= file_size
    else:
        print(
            f"{target_folder} 中的缓存文件已小于 {max_size / 1024 / 1024 / 1024} GB, 并且没有大于 {max_age / 60 / 60 / 24} 天的文件, "
            f"不需要删除.")

# folder = r'D:\stats_input'
# start_time = time.time()
# delete_old_files(folder)
# elapsed_time = time.time() - start_time
# print(f"绘图用时 {elapsed_time:.2f} 秒")

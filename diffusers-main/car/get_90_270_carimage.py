import pandas as pd
import os
car_ = pd.read_csv('Image_table.csv')
import blobfile as bf

# def _list_image_files_recursively(data_dir):
#     results = []
#     for entry in sorted(bf.listdir(data_dir)):
#         full_path = bf.join(data_dir, entry)
#         ext = entry.split(".")[-1]
#         if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
#             results.append(full_path)
#         elif bf.isdir(full_path):
#             results.extend(_list_image_files_recursively(full_path))
#     return results

#path = _list_image_files_recursively("resized_DVM")

for im in car_.iloc:
    if im[3] == 90 or im[3] == 270:
        #print([t for t in im[2].split("$$")])
        p = os.path.join(*[t for t in im[2].split("$$")])
        print(p)
        # print(im[3])
        # print(im[2])
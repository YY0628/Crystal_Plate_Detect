"""
    将没有标注记录的标注文件删除，之后训练时，仅选取有标注文件的记录
    文件格式为 xml 
"""

from xml.dom import minidom
import os

def deleteFileWithouLabel(dir):
    files = os.listdir(dir)
    for i, name in enumerate(files):
        file_name = os.path.join(dir, name)
        document = minidom.parse(open(file_name, 'r'))
        object_document = document.getElementsByTagName("object")
        if object_document == []:
            os.remove(file_name)
        
if __name__ == "__main__":
    deleteFileWithouLabel(r"H:\07-Graduation\02-datasets\imgs\labels\labels_big_dot_xml")

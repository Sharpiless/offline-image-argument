from imgaug.augmenters.geometric import Rotate
from bs4 import BeautifulSoup
import imgaug as ia
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa
import os
from numpy.core.fromnumeric import resize
from tqdm import tqdm
# xml文件生成代码
from lxml import etree

# ---- 创建标注


class CreateAnnotations:
    # ----- 初始化
    def __init__(self, flodername, filename):
        self.root = etree.Element("annotation")

        child1 = etree.SubElement(self.root, "folder")
        child1.text = flodername

        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename

        child3 = etree.SubElement(self.root, "path")
        child3.text = filename

        child4 = etree.SubElement(self.root, "source")

        child5 = etree.SubElement(child4, "database")
        child5.text = "Unknown"

    # ----- 设置size
    def set_size(self, imgshape):
        (height, witdh, channel) = imgshape
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    # ----- 保存文件
    def savefile(self, filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True,
                   xml_declaration=False, encoding='utf-8')

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        object = etree.SubElement(self.root, "object")
        namen = etree.SubElement(object, "name")
        namen.text = label
        bndbox = etree.SubElement(object, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)


if __name__ == '__main__':

    repeat_time = 3
    base = 'VOCData/dataforfen'
    outbase = 'new'
    if not os.path.exists(outbase):
        os.mkdir(outbase)

    # 增强效果
    seq = iaa.Sequential([
        iaa.GammaContrast(),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.MultiplyHue((0.5, 1.5)),
        iaa.Fliplr(0.5),
        iaa.Cutout(fill_mode="constant", cval=255),
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
    ])

    images = [v for v in os.listdir(base) if v.endswith('.jpg')]

    for img in tqdm(images[548:]):
        xml = img[:-4]+'.xml'
        if not os.path.exists(os.path.join(base, xml)):
            print(xml, 'not exist.')
            continue
        # 打开标注文件
        soup = BeautifulSoup(open(os.path.join(base, xml)), "lxml")
        # 导入图像
        image = imageio.imread(os.path.join(base, img))

        # 用于存放标注文件边界框信息
        bbsOnImg = []
        # 找到所有包含框选目标的节点
        for objects in soup.find_all(name="object"):
            # 获得当前边界框的分类名
            object_name = str(objects.find(name="name").string)
            # 提取坐标点信息
            xmin = int(objects.xmin.string)
            ymin = int(objects.ymin.string)
            xmax = int(objects.xmax.string)
            ymax = int(objects.ymax.string)
            # 保存该边界框的信息
            bbsOnImg.append(BoundingBox(x1=xmin, x2=xmax,
                                        y1=ymin, y2=ymax, label=object_name))
        # 初始化imgaug的标选框数据
        bbs = BoundingBoxesOnImage(bbsOnImg, shape=image.shape)
        for i in range(repeat_time):
            # 输入增强前的图像和边框，得到增强后的图像和边框
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
            # 创建保存类
            dst_xml = xml[:-4] + '_%4d.xml' % i
            dst_img = xml[:-4] + '_%4d.jpg' % i
            anno = CreateAnnotations(outbase, dst_img)
            anno.set_size(image_aug.shape)
            # 循环提取
            bbs_aug_clip =bbs_aug.clip_out_of_image()
            for index,bb in enumerate(bbs_aug_clip):
                xmin = int(bb.x1)
                ymin = int(bb.y1)
                xmax = int(bb.x2)
                ymax = int(bb.y2)
                label = str(bb.label)
                anno.add_pic_attr(label, xmin, ymin, xmax, ymax)
            # 保存标注文件
            anno.savefile(os.path.join(outbase, dst_xml))
            # 保存增强图像
            imageio.imsave(os.path.join(outbase, dst_img), image_aug)

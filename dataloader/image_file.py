import os
import glob
import logging
import functools
import numpy as np

class ImageFile(object):
    def __init__(self, phase='train'):
        self.logger = logging.getLogger("Logger")
        self.phase = phase
        self.rng = np.random.RandomState(0)

    def _get_valid_names(self, *dirs, shuffle=True):
        # Extract valid names  提取有效名称
        name_sets = [self._get_name_set(d) for d in dirs]
        # for i in dirs:
        #     print(i)

        # Reduce
        def _join_and(a, b):
            return a & b

        valid_names = list(functools.reduce(_join_and, name_sets))
        if shuffle:
            self.rng.shuffle(valid_names)

        if len(valid_names) == 0:
            self.logger.error('No image valid')
        else:
            self.logger.info('{}: {} foreground/images are valid'.format(self.phase.upper(), len(valid_names)))

        return valid_names

    @staticmethod
    def _get_name_set(dir_name):
        path_list = glob.glob(os.path.join(dir_name, '*'))
        name_set = set()
        for path in path_list:
            name = os.path.basename(path)   #返回path最后的文件名
            name = os.path.splitext(name)[0]  #返回前面的文件名
            name_set.add(name)
        return name_set

    @staticmethod
    def _list_abspath(data_dir, ext, data_list):
        return [os.path.join(data_dir, name + ext)
                for name in data_list]


class ImageFileTrain(ImageFile):
    def __init__(self,
                 alpha_dir="train_alpha",
                 fg_dir="train_fg",
                 bg_dir="train_bg",
                 # alpha_ext=".jpg",
                 # fg_ext=".jpg",
                 # bg_ext=".jpg",

                 # alpha_ext=".png",
                 # fg_ext=".jpg",
                 # bg_ext=".jpg",

                 alpha_ext=".jpg",
                 fg_ext=".jpg",
                 bg_ext=".jpg",
                 ):
        super(ImageFileTrain, self).__init__(phase="train")

        self.alpha_dir  = alpha_dir
        self.fg_dir     = fg_dir
        self.bg_dir     = bg_dir
        self.alpha_ext  = alpha_ext
        self.fg_ext     = fg_ext
        self.bg_ext     = bg_ext


        self.logger.debug('Load Training Images From Folders')

        self.valid_fg_list = self._get_valid_names(self.fg_dir, self.alpha_dir) #获取文件夹中图片名字
        self.valid_bg_list = [os.path.splitext(name)[0] for name in os.listdir(self.bg_dir)]
        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_fg_list)
        self.fg = self._list_abspath(self.fg_dir, self.fg_ext, self.valid_fg_list)
        self.bg = self._list_abspath(self.bg_dir, self.bg_ext, self.valid_bg_list)

    def __len__(self):
        return len(self.alpha)


class ImageFileTest(ImageFile):
    def __init__(self,
                 alpha_dir="test_alpha",
                 merged_dir="test_merged",
                 trimap_dir="test_trimap",
                 fggt_dir="test_fggt",
                 alpha_ext=".jpg",
                 merged_ext=".png",
                 trimap_ext=".jpg",
                 fggt_ext=".jpg",
                 # alpha_ext=".png",
                 # merged_ext=".png",
                 # trimap_ext=".png",
                 # fggt_ext=".png",
                 ):
        super(ImageFileTest, self).__init__(phase="test")

        self.alpha_dir  = alpha_dir
        self.merged_dir = merged_dir
        self.trimap_dir = trimap_dir
        self.alpha_ext  = alpha_ext
        self.merged_ext = merged_ext
        self.trimap_ext = trimap_ext
        self.fggt_ext = fggt_ext
        self.fggt_dir = fggt_dir

        self.logger.debug('Load Testing Images From Folders')

        self.valid_image_list = self._get_valid_names(self.alpha_dir, self.merged_dir, self.trimap_dir, shuffle=False)

        self.alpha = self._list_abspath(self.alpha_dir, self.alpha_ext, self.valid_image_list)
        self.merged = self._list_abspath(self.merged_dir, self.merged_ext, self.valid_image_list)
        self.trimap = self._list_abspath(self.trimap_dir, self.trimap_ext, self.valid_image_list)
        self.fggt = self._list_abspath(self.fggt_dir, self.fggt_ext, self.valid_image_list)

    def __len__(self):
        return len(self.alpha)


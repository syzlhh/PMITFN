import torch.utils.data as data
from glob2 import glob
import random
from PIL import Image,ImageOps
import os
import torchvision.transforms as transforms

class CityDataset2(data.Dataset):
    def __init__(self, data_dir, data_dir_clean,data_trans=None, test_hazy_dir=None, test_gt_dir=None, istrain=True, flip=True):
        super(CityDataset2, self).__init__()
        # self.hazy = hazy_img_path
        # self.clear = clear_path
        # self.unhazy = unhazy_path
        self.scale_size = 286
        self.size = 256
        self.hazy_img_list = []
        self.trans_list = []
        self.clean_img_list = []
        self.un_clean_img_list = glob('D:\OTS_BETA\clear\clear\*.jpg')
        self.isTrain = istrain
        self.Flip = flip
        if self.isTrain:
            with open('D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\train_list.txt', 'r') as f:
                lines = f.readlines()
                random.shuffle(lines)
                for img in lines:
                    img = img.strip('\n')
                    hazy = img.split(' ')[0]
                    gt = img.split(' ')[-1]
                    trans = hazy.replace("_foggy_beta", "_transmittance_beta")
                    if os.path.exists(data_trans+trans):
                        self.trans_list.append(data_trans+trans)
                        self.hazy_img_list.append(data_dir + hazy)
                        # name = clean_data_dir+img.split("_")[0]+'.jpg'
                        self.clean_img_list.append(data_dir_clean + gt)
                    else:
                        continue
        else:
            with open('D:\Data\Data\leftImg8bit_trainvaltest_foggy\leftImg8bit_foggy\\val_list.txt', 'r') as f:
                lines = f.readlines()
                for img in lines:
                    img = img.strip('\n')
                    hazy = img.split(' ')[0]
                    gt = img.split(' ')[-1]

                    self.hazy_img_list.append(data_dir + hazy)
                    # name = clean_data_dir+img.split("_")[0]+'.jpg'
                    self.clean_img_list.append(data_dir_clean + gt)


    def name(self):
        return 'BaseDataset'


    def initialize(self, opt):
        pass


    def __getitem__(self, index):
        if self.isTrain:
            hazy_img = Image.open(self.hazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')
            trans_img = Image.open(self.trans_list[index]).convert('L')

            if random.uniform(0, 1) > 0.5:
                hazy_img = hazy_img.transpose(Image.FLIP_LEFT_RIGHT)
                clean_img = clean_img.transpose(Image.FLIP_LEFT_RIGHT)
                trans_img = trans_img.transpose(Image.FLIP_LEFT_RIGHT)

            w_s, h_s = hazy_img.size
            hazy_img = hazy_img.resize((768, 512), Image.BICUBIC)
            clean_img = clean_img.resize((768, 512), Image.BICUBIC)
            trans_img = trans_img.resize((768, 512), Image.BICUBIC)

            w_s, h_s = hazy_img.size

            w_offset_s = random.randint(0, max(0, w_s - self.size * 2 - 1))
            h_offset_s = random.randint(0, max(0, h_s - self.size - 1))

            hazy_img = transforms.ToTensor()(hazy_img)
            clean_img = transforms.ToTensor()(clean_img)
            trans_img = transforms.ToTensor()(trans_img)


            hazy_img = hazy_img[:, h_offset_s:h_offset_s + self.size, w_offset_s:w_offset_s + self.size * 2]
            clean_img = clean_img[:, h_offset_s:h_offset_s + self.size, w_offset_s:w_offset_s + self.size * 2]
            trans_img = trans_img[:, h_offset_s:h_offset_s + self.size, w_offset_s:w_offset_s + self.size * 2]

            hazy_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(hazy_img)
            clean_img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(clean_img)
            trans_img = trans_img * 2 - 1

            return hazy_img, clean_img, trans_img

        else:
            hazy_img = Image.open(self.hazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')
            hazy_img = hazy_img.resize((512, 256), Image.BICUBIC)
            clean_img = clean_img.resize((512, 256), Image.BICUBIC)

            transform_list = []
            transform_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
            trans = transforms.Compose(transform_list)

            return trans(hazy_img), trans(clean_img)


    def __len__(self):
        return len(self.hazy_img_list)
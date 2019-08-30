#!/usr/bin/env python
# -*- coding:utf-8 -*-

# img_attr = dict({
#     'path': img_path (str),
#     'md5': img_md5 (str),
#     'class': img_class (str),
#     'hash': img_hash (int),
#     'file_name': img_file_name (str)
#     'is_sample': is_sample (int)
# })


import pickle
import argparse
from PIL import Image
import os
from shutil import copyfile
import time
import imagehash


# hash编码二制位数对照
hash_bit_dict = dict({
    'dhash': 128,
    'phash': 64
})


def cal_hamming_dist(hash1, hash2):
    """Calculate number of bits different between two hashes.
    >>> get_num_bits_different(0x4bd1, 0x4bd1)
    0
    >>> get_num_bits_different(0x4bd1, 0x5bd2)
    3
    >>> get_num_bits_different(0x0000, 0xffff)
    16
    """
    return bin(hash1 ^ hash2).count('1')


def cal_hash(img_path, hash_type):
    """计算img_path图片的hash

     Image.open 在加载图片可能出错，建议在调用此函数时使用 try...except...

    Args:
        img_path: 图片的绝对路径
        hash_type: hash 类型。目前可选的为 phash 、 dhash 。

    Returns:
        long类型。hash 的十进制值
    """
    img = Image.open(img_path)
    if hash_type == 'dhash':
        img_hash_h = str(imagehash.dhash(img))
        img_hash_v = str(imagehash.dhash_vertical(img))
        img_hash = int(img_hash_h+img_hash_v, 16)
    elif hash_type == 'phash':
        img_hash = int(str(imagehash.phash(img)), 16)
    else:
        print('Error invalid hash_type: %s' % (hash_type))
        exit(-1)

    return img_hash


def split_hash(hash_int, seg_count, seg_len):
    """将hash_int值按二进制分割为seg_count个片段，每个片段长度为seg_len

    将hash值
        0xe8747c3f3239b4e28801c04c6f3f3000
    切割为4个片段, 每个片段长度为32bits，
        [0xe8747c3f, 0x3239b4e2, 0x8801c04c, 0x6f3f3000]
    如果有多余从高位开始丢弃多余的位。如将上文中的hash值切割为5个片段，每个片段长度为24bits，
        [0x747c3f, 0x3239b4, 0xe28801, 0xc04c6f, 0x3f3000]
    如果有不足在高位之前补0，如将上文中的hash值切割为5个片段，每个片段长度为32bits，
        [0x0, 0xe8747c3f, 0x3239b4e2, 0x8801c04c, 0x6f3f3000]

    Args:
        hash_int: 十进制的hash值。int类型
        seg_count: 切割的片数。int类型
        seg_len: 片段的二进制长度。int类型

    Returns:
        list类型。分割后的片段list，每个片段为十进制的hash值
    """
    if seg_len <= 0:
        print('Error split_hash: invalid seg_len %d' % (seg_len))
        exit(-1)
        
    hash_segs = []
    for i in range(seg_count):
        hash_seg = hash_int % (2 ** seg_len)
        hash_int /= 2 ** seg_len
        hash_segs.append(hash_seg)
    # 变为大端
    hash_segs.reverse()

    return hash_segs


def load_hash_list(hash_list_path):
    """加载图片的hash值

    加载图片的标签类别，md5值，hash值。其中hash值读取为10进制数。

    Args:
        hash_list_path: hash_list 的加载路径。str类型

    Returns:
        dict类型。 img_md5 --> img_attr ('class', 'md5', 'hash')
        example:
        {'b14464e023427a3c912fe95096679f93':
         {'class': 'porn' ,'md5': 'b14464e023427a3c912fe95096679f93', 'hash': 1d9d9a64860323076f7dbb40803e0307},
         ...}
    """
    imgs_dict = dict()
    with open(hash_list_path, 'r') as hash_file:
        # 跳过标题
        hash_file.readline()
        for attr in hash_file.readlines():
            attr = attr.strip()
            img_class = attr.split(',')[0]
            img_md5 = attr.split(',')[1]
            img_hash = int(attr.split(',')[2], 16)
            if img_md5 in imgs_dict:
                print('Error same md5 img1[class:%s, md5:%s, hash:%s] img2[class:%s, md5:%s, hash:%s]' %
                      (img_class, img_md5, img_hash,
                       imgs_dict[img_md5]['class'],
                       imgs_dict[img_md5]['md5'],
                       imgs_dict[img_md5]['hash']))
            else:
                imgs_dict[img_md5] = dict({'class': img_class,
                                           'md5': img_md5,
                                           'hash': img_hash})
    return imgs_dict


def save_hash_list(imgs_dict, hash_list_path, hash_bit):
    """保存图片的hash值

    保存图片的标签类别，md5值，hash值。其中hash值以hash_bit位16进制的形式保存。

    Args:
        imgs_dict: img_attr 字典，img_md5 --> img_attr。
        hash_list_path: hash_list 的保存路径。str类型
        hash_bit: hash 值的十六进制位数。int类型
    """
    with open(hash_list_path, 'w') as hash_file:
        hash_file.write('class,md5,hash\n')
        for img_md5, img_attr in imgs_dict.items():
            # 其中 hash 以 hash_bit 位十六进制形式保存
            img_class = img_attr['class']
            img_md5 = img_attr['md5']
            img_hash = img_attr['hash']
            save_format = '%s,%s,%0' + str(hash_bit) + 'x,\n'
            hash_file.write(save_format % (img_class, img_md5, img_hash))
    print('save %d imgs hash_list to %s' % (len(imgs_dict), hash_list_path))


def load_dataset(dataset_path, flags):
    """从 dataset 中加载图片，返回图片属性列表。

    dataset 的目录结构应该为
        <dataset_name>
        ├── [<hash_type>.list]
        ├── <class_name1>
        │   ├── 000448b0c6fff7e7c27526476a9da88e.jpg
        |   └── ...
        ├── <class_name2>
        │   ├── 021c36b5231f23ba68d5f8cd7ae55d82.jpg
        |   └── ...
        └── <class_name3>
            ├── 000448b0c6fff7e7c27526476a9da88e.jpg
            └── ...

    Args:
        dataset_path: 数据集路径。string类型
        flags: 参数。其中 hash_type 、 is_load_hash_list 、is_save_hash_list 是必须的。
    Returns:
        list 类型。图片属性列表。
        example:
        [{'class': img_class, 'md5': img_md5, 'hash': img_hash,
          'file_name': img_file_name, 'path', img_path}, ...]
    """
    if not os.path.exists(dataset_path):
        print('Error dataset %s not exists'
              % (dataset_path))
        return []
    else:
        print('Load dataset from %s' % (dataset_path))

    hash_list_file_name = flags.hash_type + '.list'
    hash_list_path = os.path.join(dataset_path, hash_list_file_name)
    is_cal_hash = 1
    if flags.is_load_hash_list:
        if os.path.exists(hash_list_path):
            imgs_dict = load_hash_list(hash_list_path)
            is_cal_hash = 0
        else:
            imgs_dict = dict()
    else:
        imgs_dict = dict()

    dataset_walk_list = list(os.walk(dataset_path))
    imgs_list = []
    for class_idx, class_dir in enumerate(dataset_walk_list[0][1]):
        for img_file_name in dataset_walk_list[class_idx+1][2]:
            img_path = os.path.join(dataset_path, class_dir, img_file_name)
            img_md5 = img_file_name.split('.')[0]
            img_class = class_dir
            # load img_hash
            try:
                if is_cal_hash:
                    hash_type = flags.hash_type
                    img_hash = cal_hash(img_path, hash_type)
                    imgs_dict[img_md5] = dict({'class': img_class,
                                               'md5': img_md5,
                                               'hash': img_hash})
                else:
                    if img_md5 in imgs_dict:
                        img_hash = imgs_dict[img_md5]['hash']
                    else:
                        print('%s is lack of hash' % (img_path))  # debug
                        hash_type = flags.hash_type
                        img_hash = cal_hash(img_path, hash_type)
                        imgs_dict[img_md5] = dict({'class': img_class,
                                                   'md5': img_md5,
                                                   'hash': img_hash})
            except Exception as e:
                print('Error cal_hash img_path: %s error: %s' % (img_path, e))
                continue

            img_attr = imgs_dict[img_md5]
            img_attr['file_name'] = img_file_name
            img_attr['path'] = img_path
            imgs_list.append(img_attr)

    hash_bit = hash_bit_dict[flags.hash_type]
    if flags.is_save_hash_list:
        try:
            save_hash_list(imgs_dict, hash_list_path, hash_bit)
        except Exception as e:
            print('Error save_hash_list %s' % (e))

    return imgs_list


def load_dataset_list(list_path, flags):
    """从数据集列表中加载图片，返回图片属性列表。

    Args:
        list_path: 数据集列表路径。string类型
        flags: 参数。其中 hash_type 、 is_load_hash_list 、is_save_hash_list 是必须的。
    Returns:
        list 类型。图片属性列表。
        example:
        [{'class': img_class, 'md5': img_md5, 'hash': img_hash,
          'file_name': img_file_name, 'path', img_path}, ...]
    """
    if not os.path.exists(list_path):
        print('Error dataset list: %s not exists' % (list_path))
        exit(-1)
    else:
        print('Load dataset list from %s' % (list_path))

    imgs_list = []
    with open(list_path, 'r') as list_file:
        for dataset_path in list_file.readlines():
            dataset_path = dataset_path.strip()
            imgs_list += load_dataset(dataset_path, flags)

    return imgs_list


def init_hash_segs_table(imgs_list, seg_count, hash_bit):
    """建立hash片段索引表

    hash_segs_table 为hash片段索引字典，每张图片根据其每个hash片段的值，将其归为索引字典的对应key下。
    例如某图片的hash片段为
        [0xe8747c3f, 0x3239b4e2, 0x8801c04c, 0x6f3f3000]
    则该图片将分别归类到
        hash_segs_table[0xe8747c3f]
        hash_segs_table[0x3239b4e2]
        hash_segs_table[0x8801c04c]
        hash_segs_table[0x6f3f3000]

    Args:
        imgs_list: 图片属性列表。list类型
            example:
            [img_attr, ...]
        seg_count: 切割的片数。int类型
        hash_bit: hash 值的二进制位数。int类型

    Returns:
        一个字典，字典的每个key是一个hash片段值，对应的item是有该片段的图片的list
        example:
        {0xe8747c3f: [img_attr, ...], 0x3239b4e2: [img_attr, ...], ...}

    """
    hash_segs_table = dict()
    for img_attr in imgs_list:
        img_hash_segs = split_hash(img_attr['hash'],
                                   seg_count,
                                   hash_bit/seg_count)
        for img_hash_seg in img_hash_segs:
            if img_hash_seg not in hash_segs_table:
                hash_segs_table[img_hash_seg] = [img_attr]
            else:
                hash_segs_table[img_hash_seg].append(img_attr)

    return hash_segs_table


def search_hash_segs_table(src_img_attr,
                           hash_segs_table,
                           hash_bit,
                           seg_count,
                           hamming_dist_thr):
    """搜索与传入图片相似相似的图片

    返回的相似图片，按相似度从高到低排序，中包含其本身

    Args:
        src_img_attr: 原图片的属性组
        hash_segs_table: hash 片段索引表
        hash_bit: hash 值的二进制位数。int类型
        seg_count: 切割的片数。int类型
        hamming_dist_thr: 判断图片相似的hamming距离的阈值

    Returns:
        list 类型。与原图片相似图片的属性列表
        example:
        [img_attr, ...]
    """
    hash_segs = split_hash(src_img_attr['hash'], seg_count, hash_bit/seg_count)
    # 根据 hash 片段生成搜索空间，避免全局搜索
    search_imgs_list = []
    for seg in hash_segs:
        search_imgs_list.extend(hash_segs_table[seg])
    similar_imgs_list = []
    for img_attr in search_imgs_list:
        hamming_dist = \
            cal_hamming_dist(src_img_attr['hash'], img_attr['hash'])
        # is similar img
        if hamming_dist <= hamming_dist_thr:
            # remove duplicates
            if img_attr not in similar_imgs_list:
                similar_imgs_list.append(img_attr)
        # save hamming_dist
        # 这两个 attr 只对当前次 search 有效。
        img_attr['hamming_cmp'] = \
            dict({'cmp_img_md5': src_img_attr['md5'],
                  'hamming_dist': hamming_dist})

    # sort by hamming_dist from low to high
    similar_imgs_list.sort(key=lambda k: k['hamming_cmp']['hamming_dist'])

    return similar_imgs_list


def hash_sample(imgs_list, flags):
    for img_attr in imgs_list:
        img_attr['is_sample'] = 1

    hamming_dist_thr = flags.hamming_dist_thr
    # 抽屉原理
    # 假设判定两张图的汉明距离小于等于 3 为相似。则将两者图的 hash 值用相同
    # 的方式切成 4 段，那么如果两者图相似，那么他们至少有一个 hash 段相同。
    seg_count = hamming_dist_thr + 1
    hash_bit = hash_bit_dict[flags.hash_type]
    # init hash segs table
    hash_segs_table = init_hash_segs_table(imgs_list, seg_count, hash_bit)

    sim_families_list = []
    processed = 0
    for img_attr in imgs_list:
        processed += 1
        if processed % 1000 == 0:
            print('Processed imgs: ', processed)
        if img_attr['is_sample'] == 0:
            continue
        # 与此图片相似的图片
        similar_imgs_list = search_hash_segs_table(img_attr,
                                                   hash_segs_table,
                                                   hash_bit,
                                                   seg_count,
                                                   hamming_dist_thr)
        sim_families_list.append(similar_imgs_list)
        dirty_list = []
        for sim_img_attr in similar_imgs_list:
            if sim_img_attr['is_sample'] != 0:
                sim_img_attr['is_sample'] = 0
            else:
                dirty_list.append(sim_img_attr)
        for dirty_img_attr in dirty_list:
            similar_imgs_list.remove(dirty_img_attr)
        # 相似图片类中选取原图供采样
        img_attr['is_sample'] = 1

    # sort by family size from big to small
    sim_families_list.sort(key=lambda k: len(k), reverse=True)
    if flags.is_remove_similar_img:
      sim_families_list_new = []
      for img_attr_list in sim_families_list:
        sim_families_list_new.append([img_attr_list[0]])
      print("图片总量: {}\tphash去重后图片总量: {}".format(len(imgs_list), len(sim_families_list_new)))
      sim_families_list = sim_families_list_new
    return sim_families_list


def hash_find(imgs_list, flags):
    # load src img
    if not os.path.exists(flags.src_img_path):
        print('Error hash find: %s not exists' % (flags.src_img_path))
        exit(-1)
    src_img_md5 = flags.src_img_path.split('/')[-1].split('.')[0]
    src_img_attr = dict({'path': flags.src_img_path, 'md5': src_img_md5})
    try:
        hash_type = flags.hash_type
        src_img_hash = cal_hash(src_img_attr['path'], hash_type)
    except Exception as e:
        print('Error cal_hash img_path: %s error: %s'
              % (src_img_attr['path'], e))
        exit(-1)
    src_img_attr['hash'] = src_img_hash

    hamming_dist_thr = flags.hamming_dist_thr
    seg_count = hamming_dist_thr + 1
    hash_bit = hash_bit_dict[flags.hash_type]
    hash_segs_table = init_hash_segs_table(imgs_list, seg_count, hash_bit)

    # 与此图片相似的图片
    similar_imgs_list = search_hash_segs_table(src_img_attr,
                                               hash_segs_table,
                                               hash_bit,
                                               seg_count,
                                               hamming_dist_thr)

    return similar_imgs_list


def save_imgs_list(imgs_list, save_path, save_attr):
    with open(save_path, 'w') as list_file:
        for img_attr in imgs_list:
            for attr_name in save_attr:
                list_file.write('%s,' % (str(img_attr[attr_name])))
            list_file.write('\n')


def save_sample_list(imgs_list, save_path, save_attr):
    sample_count = 0
    clean_count = 0
    with open(save_path, 'w') as list_file:
        for img_attr in imgs_list:
            if img_attr['is_sample'] == 0:
                clean_count += 1
            else:
                sample_count += 1
            for attr_name in save_attr:
                list_file.write('%s,' % (str(img_attr[attr_name])))
            list_file.write('\n')

    return sample_count, clean_count


def load_sample_dict(sample_list_path, sample_dict=None):
    if not os.path.exists(sample_list_path):
        print('Error family list %s not exists' % (sample_list_path))
        exit(-1)

    if sample_dict is None:
        sample_dict = dict()

    with open(sample_list_path, 'r') as sample_list_f:
        for line in sample_list_f.readlines():
            img_md5 = line.split(',')[0]
            img_is_sample = int(line.split(',')[1])
            if img_md5 in sample_dict:
                print('Error same md5: %s' % (img_md5))
            else:
                sample_dict[img_md5] = img_is_sample

    return sample_dict


def copy_imgs_list(dest_dir, imgs_list):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    print('Copy %d imgs to %s' % (len(imgs_list), dest_dir))
    for img_attr in imgs_list:
        img_file_name = os.path.basename(img_attr['path'])
        copyfile(img_attr['path'],
                 os.path.join(dest_dir, img_file_name))


def statistics(family_list, isolated_img_count):
    pass


def clean_mode(flags):
    print('clean mode')
    # calculate hash
    print('load data')
    start = time.clock()
    imgs_list = load_dataset_list(flags.dataset_list_path, flags)
    elapsed = (time.clock() - start)
    print('load %d imgs' % (len(imgs_list)))
    print('load data elapsed: %f' % (elapsed))

    # hash sample
    print('hash_sample')
    start = time.clock()
    sim_families_list = hash_sample(imgs_list, flags)
    elapsed = (time.clock() - start)
    print('hash_sample elapsed: %f' % (elapsed))

    # save sample list
    dataset_list_name = \
        os.path.basename(flags.dataset_list_path).split('.')[0]
    sample_list_file_name = dataset_list_name + '.sample.list'
    sample_list_dir = './'
    sample_list_path = os.path.join(sample_list_dir, sample_list_file_name)
    # 存储的位置可能没权限
    try:
        save_sample_list(imgs_list, sample_list_path, ['md5', 'is_sample'])
    except Exception as e:
        print('Error save_family_list %s' % (e))

    # save similar families
    sim_list_save_name = os.path.basename(flags.dataset_list_path).split('.')[0] + "_sim.list"
    with open(sim_list_save_name, 'w') as f:
        for idx, fam_imgs_list in enumerate(sim_families_list):
            for img_attr in fam_imgs_list:
                f.write("{}\t{}\n".format(idx, img_attr['path']))
    if flags.sim_fams_save_path != '0':
        for idx, fam_imgs_list in enumerate(sim_families_list):
            dst_path = os.path.join(flags.sim_fams_save_path, str(idx))
            os.makedirs(dst_path)
            copy_imgs_list(dst_path, fam_imgs_list)


def find_mode(flags):
    print('find mode')
    # calculate hash
    print('load data')
    start = time.clock()
    imgs_list = load_dataset_list(flags.dataset_list_path, flags)
    elapsed = (time.clock() - start)
    print('load %d imgs' % (len(imgs_list)))
    print('load data elapsed: %f' % (elapsed))

    print('find similar imgs')
    start = time.clock()
    similar_imgs_list = hash_find(imgs_list, flags)
    elapsed = (time.clock() - start)
    print('find %d similar imgs' % (len(similar_imgs_list)))
    print('find similar imgs elapsed: %f' % (elapsed))

    # save list
    src_img_md5 = os.path.basename(flags.src_img_path).split('.')[0]
    sim_list_file_name = src_img_md5 + '.similar.imgs.list'
    sim_list_dir = './'
    sim_list_path = os.path.join(sim_list_dir, sim_list_file_name)
    save_imgs_list(similar_imgs_list, sim_list_path, ['path'])

    # save imgs
    if flags.sim_imgs_save_path != '0':
        copy_imgs_list(flags.sim_imgs_save_path, similar_imgs_list)


def label_task_mode(flags):
    print('label task mode')
    # calculate hash
    print('load data')
    start = time.clock()
    imgs_list = load_dataset(flags.dataset_path, flags)
    elapsed = (time.clock() - start)
    print('load %d imgs' % (len(imgs_list)))
    print('load data elapsed: %f' % (elapsed))

    # hash sample
    print('hash_sample')
    start = time.clock()
    sim_families_list = hash_sample(imgs_list, flags)
    elapsed = (time.clock() - start)
    print('hash_sample elapsed: %f' % (elapsed))

    # save sim_families_list
    if flags.sim_families_list_path is None:
        print('save path of sim_families_list is empty')
        exit(-1)
    else:
        print('save sim_families_list to %s' % (flags.sim_families_list_path))
    with open(flags.sim_families_list_path, 'wb') as var_file:
        pickle.dump(sim_families_list, var_file)


def add_argument(parser):
    parser.add_argument('--mode', type=str, default='clean',
                        help='模式。有 clean 模式、 find 模式 、 label_task 模式')
    # 数据集参数
    parser.add_argument('--dataset_list_path', type=str, default=None,
                        help='dataset_list的加载路径')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='dataset的加载路径')
    # hash 参数
    parser.add_argument('--hash_type', type=str, default='phash',
                        help='hash 类型。现在可选的有 phash 、 dhash 。默认为 phash')
    parser.add_argument('--hamming_dist_thr', type=int, default=3,
                        help='判断相似的hamming距离阈值。默认为 3。')
    parser.add_argument('--is_load_hash_list', type=int, default=1,
                        help='是否加载数据目录下的hash_list，加载的文件为 <hash_type>.list 。默认为 1。')
    parser.add_argument('--is_save_hash_list', type=int, default=0,
                        help='是否在数据目录下保存hash_list，保存会覆盖，保存为 <hash_type>.list 。默认为 0。')
    # clean mode 参数
    parser.add_argument('--sim_fams_save_path', type=str, default='0',
                        help='clean 模式，保存相似的图片类的文件夹路径')
    # find mode 参数
    parser.add_argument('--src_img_path', type=str, default=None,
                        help='find 模式，用于搜索的图片的路径')
    parser.add_argument('--sim_imgs_save_path', type=str, default='0',
                        help='find 模式，保存相似的图片的文件夹路径')
    # label_task_mode
    parser.add_argument('--sim_families_list_path', type=str, default=None,
                        help='label_task 模式，保存 sim_families_list 变量的路径')
    parser.add_argument('--is_remove_similar_img', type=int, default=0,
                        help='clean 模式，相似图片仅保留一张')


def main(flags):
    if flags.mode == 'clean':
        clean_mode(flags)
    elif flags.mode == 'find':
        find_mode(flags)
    elif flags.mode == 'label_task':
        label_task_mode(flags)
    else:
        print('Error invalid mode %s' % (flags.mode))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    if len(unparsed) > 0:
        print("there are unknow args %s " % ','.join(unparsed))
        exit(-1)

    main(FLAGS)

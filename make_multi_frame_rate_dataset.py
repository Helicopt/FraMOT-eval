import re
import os
import configparser
import argparse
from senseTk.common import Det, TrackSet
from PIL import Image

skips = [1, 2, 4, 8, 16, 25, 36, 50, 75]
# seqs = [('MOT20-%02d' % i, 'train') for i in [1, 2, 3, 5]] + \
#     [('MOT20-%02d' % i, 'test') for i in [4, 6, 7, 8]]
src_dir = '/mnt/lustre/share/fengweitao/MOT20/test'
dst_dir = '/mnt/lustre/share/fengweitao/MOT20/mfr_test/'


def generate_indices(frames, skip=1, separate=False):
    n = len(frames)
    ret = []
    r = n % skip
    reverse = False
    for i in range(skip):
        tmp = [(j * skip + i) for j in range(n // skip + (i < r))]
        if separate:
            ret.append([frames[j] for j in tmp])
        else:
            if reverse:
                ret.extend(tmp[::-1])
            else:
                ret.extend(tmp)
            reverse = not reverse
    if not separate:
        ret = [[frames[j] for j in ret]]
    v = set()
    for one in ret:
        v = v.union(one)
    assert len(v) == n, '%d vs %d' % (len(v), n)
    return ret


def generate(frames, ground_truth, seq_dir, image_height, image_width, skip, virtual_seqname):
    formatter = '{root}/{seq}/img1/{fr}.{ext}'
    # all_data = []
    ts = TrackSet()
    os.makedirs(os.path.join(dst_dir, virtual_seqname, 'img1'), exist_ok=True)
    for i, frame_id in enumerate(frames):
        img = os.path.join(seq_dir, 'img1', '%06d.jpg' % frame_id)
        dst_path = os.path.join(dst_dir, virtual_seqname,
                                'img1', '%06d.jpg' % (i + 1))
        if os.path.exists(dst_path):
            os.unlink(dst_path)
        os.symlink(img, dst_path)
        # data = {
        #     'filename': os.path.abspath(img),
        #     'instances': ground_truth[frame_id] if ground_truth else None,
        #     'formatter': formatter,
        #     'virtual_filename': os.path.join('/', 'virtual_path', virtual_seqname, '%06d.jpg' % (i + 1))
        # }
        # all_data.append(data)
        if ground_truth:
            for d in ground_truth[frame_id]:
                d = d.copy()
                d.fr = i + 1
                ts.append_data(d)
    os.makedirs(os.path.join(dst_dir, virtual_seqname, 'gt'), exist_ok=True)
    if ground_truth:
        with open(os.path.join(dst_dir, virtual_seqname, 'gt', 'gt.txt'), 'w') as fd:
            ts.dump(fd, formatter='fr.i,id.i,x1,y1,w,h,st.i,la.i,cf')
    iniconfig = configparser.ConfigParser()
    iniconfig.add_section('Sequence')
    iniconfig.set('Sequence', 'seqLength', str(len(frames)))
    iniconfig.set('Sequence', 'frameRate', '%.2f' % (25. / skip))
    iniconfig.set('Sequence', 'name', virtual_seqname)
    iniconfig.set('Sequence', 'imDir', 'img1')
    iniconfig.set('Sequence', 'imExt', '.jpg')
    iniconfig.set('Sequence', 'imWidth', str(image_width))
    iniconfig.set('Sequence', 'imHeight', str(image_height))
    with open(os.path.join(dst_dir, virtual_seqname, 'seqinfo.ini'), 'w') as fd:
        iniconfig.write(fd)
    # return all_data


def convert_annos(half=False, gap=30, separate=False):
    '''
    convert mot annotations
    '''
    seqs = os.listdir(src_dir)
    for seq in seqs:
        print('processing', seq)
        seq_dir = os.path.join(src_dir, seq)
        if not os.path.isdir(seq_dir):
            continue
        gt_path = os.path.join(seq_dir, 'gt', 'gt.txt')
        if os.path.exists(gt_path):
            ground_truth = TrackSet(gt_path)
        else:
            ground_truth = None
        image_probe = Image.open(os.path.join(seq_dir, 'img1', '000001.jpg'))
        image_height, image_width = image_probe.height, image_probe.width
        inifile = os.path.join(seq_dir, 'seqinfo.ini')
        iniconfig = configparser.ConfigParser()
        iniconfig.read(inifile)
        min_fr = 1
        max_fr = int(iniconfig['Sequence']['seqLength'])
        original_fps = float(iniconfig['Sequence']['frameRate'])
        suffix = '_multi_framerate'
        if separate:
            suffix += '_sep'
        frames = list(range(min_fr, max_fr + 1))
        half_ = (min_fr + max_fr) // 2
        for skip in skips:
            print('-- processing', '%.1f' % (25 / skip), 'fps')
            if half:
                ind_list0 = generate_indices(
                    [fr for fr in frames if fr <= half_], skip=skip, separate=separate)
                ind_list1 = generate_indices(
                    [fr for fr in frames if fr > half_ + gap], skip=skip, separate=separate)
                for i, indices in enumerate(ind_list0):
                    if separate:
                        virtual_seqname_0 = 'S-%d-%d-%s-ht' % (skip, i, seq)
                    else:
                        virtual_seqname_0 = 'S-%d-%s-ht' % (skip, seq)
                    generate(indices, ground_truth,
                             seq_dir, image_height, image_width, skip, virtual_seqname_0)
                for i, indices in enumerate(ind_list1):
                    if separate:
                        virtual_seqname_1 = 'S-%d-%d-%s-hv' % (skip, i, seq)
                    else:
                        virtual_seqname_1 = 'S-%d-%s-hv' % (skip, seq)
                    generate(indices, ground_truth,
                             seq_dir, image_height, image_width, skip, virtual_seqname_1)
            indices_list = generate_indices(
                frames, skip=skip, separate=separate)
            for i, indices in enumerate(indices_list):
                if separate:
                    virtual_seqname = 'S-%d-%d-%s' % (skip, i, seq)
                else:
                    virtual_seqname = 'S-%d-%s' % (skip, seq)
                generate(indices, ground_truth,
                         seq_dir, image_height, image_width, skip, virtual_seqname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=None,
                        help='the source dataset dir, where sequence dirs are located')
    parser.add_argument('--dst', type=str, default=None,
                        help='the destination dataset dir, where the multi-frame-rate versions will be placed')
    parser.add_argument('--half', default=False, action='store_true',
                        help='split the half dataset')
    args = parser.parse_args()
    if args.src is not None:
        src_dir = args.src
    if args.dst is not None:
        dst_dir = args.dst
    convert_annos(half=args.half, separate=True)

import argparse
from cleanfid import fid

from base_dataset import BaseDataset
from utils import inception_score, perceptual_distance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')
   
    ''' parser configs '''
    args = parser.parse_args()
    
    # Metric FID
    fid_score = fid.compute_fid(args.src, args.dst, device='cpu', num_workers=0)
    print('FID: {}'.format(fid_score))

    # Inception Score
    is_mean, is_std = inception_score(BaseDataset(args.dst), cuda=False, batch_size = 1, resize=True)
    print('IS: {} {}'.format(is_mean, is_std))

    # Perceptual Distance
    pd = perceptual_distance(BaseDataset(args.src), BaseDataset(args.dst), cuda=False, batch_size = 1, resize=True)
    print('PD: {}'.format(pd))



import argparse
class TrainOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='../IRVI/Data/MSRS1', help='path of data')
    self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=16, help='# of threads for data loader')

    # training related
    self.parser.add_argument('--lr', default=1e-4, type=int, help='Initial learning rate for training model')
    self.parser.add_argument('--n_ep', type=int, default=1100, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=100, help='epoch start decay learning rate, set -1 if no decay') # 200 * d_iter
    self.parser.add_argument('--resume', type=str, default=None, help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    self.parser.add_argument('--seed', type=int, default=3047, help='random seed (default: 1)')
    self.parser.add_argument('--save_path_para', type=str, default='./checkpoints/')


    # progrisive train
    self.parser.add_argument('--encoder_path', type=str, default='./results/PSFusion/checkpoints/encoder_best.pth', help='the input channel of the net')
    self.parser.add_argument('--decoder_path', type=str, default='./results/PSFusion/checkpoints/decoder_best.pth', help='the input channel of the net')

    # ouptput related
    self.parser.add_argument('--in_ch', type=int, default=1, help='the input channel of the net')
    self.parser.add_argument('--out_ch', type=int, default=1, help='the output channel of the net')
    self.parser.add_argument('--name', type=str, default='PSFusion', help='folder name to save outputs')
    self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
    self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving result images and models')
    self.parser.add_argument('--display_freq', type=int, default=10, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=10, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=50, help='freq (epoch) of saving models')
    
  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    # print('\n--- load options ---')
    # for name, value in sorted(args.items()):
    #   print('%s: %s' % (str(name), str(value)))
    return self.opt

class TestOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot', type=str, default='../IRVI/Data/TNO/', help='path of data')
    self.parser.add_argument('--dataname', type=str, default='MSRS', help='name of dataset')
    self.parser.add_argument('--phase', type=str, default='CT-MRI', help='phase for dataloading')
    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=16, help='# of threads for data loader')
    self.parser.add_argument('--save_path_para', type=str, default='./results/PSFusion/checkpoints')

    ## mode related
    self.parser.add_argument('--class_nb', type=int, default=9, help='class number for segmentation model')
    self.parser.add_argument('--resume', type=str, default='./results/PSFusion/checkpoints/best_model.pth', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--resume_en', type=str, default='./results/PSFusion/checkpoints/00199_encoder.pth', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--resume_de', type=str, default='./results/PSFusion/checkpoints/00199_decoder.pth', help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--in_ch', type=int, default=1, help='class number for segmentation model')
    self.parser.add_argument('--out_ch', type=int, default=1, help='class number for segmentation model')
    self.parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    
    # results related
    self.parser.add_argument('--name', type=str, default='PSFusion', help='folder name to save outputs')
    self.parser.add_argument('--result_dir', type=str, default='./Fusion_results', help='path for saving result images and models')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    return self.opt

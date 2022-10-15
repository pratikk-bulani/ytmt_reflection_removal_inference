### Dataset ###
from random import shuffle
import torch, os
from PIL import Image
from data.transforms import to_tensor

class ImageTestDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, finetune=False, if_align=False):
        super(ImageTestDataset, self).__init__()
        self.datadir = datadir
        self.fns = os.listdir(datadir)
        self.finetune = finetune
        self.if_align = if_align

    def align(self, x1):
        h, w = x1.height, x1.width
        h, w = h // 16 * 16, w // 16 * 16
        x1 = x1.resize((w, h))
        return x1

    def __getitem__(self, index):
        fn = self.fns[index]

        m_img = Image.open(os.path.join(self.datadir, fn)).convert('RGB')

        if self.if_align:
            m_img = self.align(m_img)

        M = to_tensor(m_img)

        # dic = {'input': M, 'target_t': None, 'fn': fn, 'real': True, 'target_r': None,
            #    'identity': self.finetune, 'identity_r': False}  # fake reflection gt
        dic = {'input': M, 'fn': fn, 'real': True,
               'identity': self.finetune, 'identity_r': False}  # fake reflection gt
        return dic

    def __len__(self):
        return len(self.fns)

### Options ###
from options.net_options.train_options import TrainOptions
import torch.backends.cudnn as cudnn

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

opt.inet = 'ytmt_ucs_old'
opt.model = 'twostage_ytmt_model'
opt.name = 'ytmt_uct_sirs_test'
opt.hyper = True
opt.if_align = True
opt.resume = True
opt.icnn_path = './checkpoints/ytmt_uct_sirs_test/ytmt_uct_sirs_68_077_00595364.pt'
if opt.dataset_path == "": raise Exception("Error: Please provide --dataset_path")
if opt.output_path == "": raise Exception("Error: Please provide --output_path")
os.makedirs(opt.output_path, exist_ok=True)

### Dataset creation and Dataloader ###
test_dataset = ImageTestDataset(opt.dataset_path, if_align=opt.if_align)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

### Test the image ###
from engine import Engine
engine = Engine(opt)
engine.test(test_dataloader, savedir=opt.output_path)
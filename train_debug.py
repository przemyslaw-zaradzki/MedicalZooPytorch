import os
import numpy as np
import torch
from torch.utils.data import Dataset

import lib.utils as utils
from lib.medloaders.medical_loader_utils import create_oct_sub_volumes



class MMS3D(Dataset):
    """
    """
    
    def __init__(
            self,
            mode,
            crop_dim,
            full_vol_dim,
            samples_per_scan,
            root,
            scan_dir_path,
            labels_dir_path,
            split_id,
            augmentation=False,
            load=False
        ):

        self.mode = mode
        self.augmentation = augmentation

        self.save_name = root + '/list-' + mode + '-samples-per-img-' + str(samples_per_scan) + '.txt'

        if augmentation:
            self.transform = None

        if load:
            print("Loading dataset")
            ## load pre-generated data
            self.list = utils.load_list(self.save_name)
            return

        subvol = '_vol_' + str(crop_dim[0]) + 'x' + str(crop_dim[1]) + 'x' + str(crop_dim[2]) + '_samples_per_scan_' + str(samples_per_scan)
        sub_vol_path = root+ '/' + mode + subvol + '/'
        utils.make_dirs(sub_vol_path)

        list_oct_3D_scans = [
            os.path.join(scan_dir_path, x) for x in sorted(os.listdir(scan_dir_path))
        ]

        list_oct_2D_labes = [
            os.path.join(labels_dir_path, x) for x in sorted(os.listdir(labels_dir_path))
        ]

        if mode=='train':
            self.list = create_oct_sub_volumes(
                list_oct_3D_scans=list_oct_3D_scans[:split_id[0]],
                list_oct_2D_labes=list_oct_2D_labes[:split_id[0]],
                mode=mode,
                samples_per_scan=samples_per_scan,
                crop_size=crop_dim,
                full_vol_dim=full_vol_dim,
                sub_vol_path=sub_vol_path
            )
        elif mode=='val':
            self.list = create_oct_sub_volumes(
                list_oct_3D_scans=list_oct_3D_scans[split_id[0]:split_id[1]],
                list_oct_2D_labes=list_oct_2D_labes[split_id[0]:split_id[1]],
                mode=mode,
                samples_per_scan=samples_per_scan,
                crop_size=crop_dim,
                full_vol_dim=full_vol_dim,
                sub_vol_path=sub_vol_path
            )
        elif mode=='test':
            pass
        elif mode=='viz':
            pass

        utils.save_list(self.save_name, self.list)


    def __len__(self):
        return len(self.list)


    def __getitem__(self, index):
        t1_path, seg_path = self.list[index]
        t1, s = np.load(t1_path), np.load(seg_path)

        if self.mode == 'train' and self.augmentation:
            print('augmentation reee')
            [augmented_t1], augmented_s = self.transform([t1], s)

            return (
                torch.FloatTensor(augmented_t1.copy()).unsqueeze(0), 
                torch.FloatTensor(augmented_s.copy())
            )

        return torch.FloatTensor(t1).unsqueeze(0), torch.FloatTensor(s)



from torch.utils.data import DataLoader



def generate_datasets(args, path='.././datasets'):
    params = {
        'batch_size': args.batchSz,
        'shuffle': args.shuffle,
        'num_workers': args.num_workers
    }
    samples_train = args.samples_train
    samples_val = args.samples_val
    split_percent = args.split

    if args.dataset_name == "MMS3D":
        train_loader = MMS3D(
            full_vol_dim=args.full_vol_dim, 
            root=args.root,
            scan_dir_path=args.scan_dir_path,
            labels_dir_path=args.labels_dir_path,
            mode='train', 
            crop_dim=args.dim,
            split_id=args.split_id, 
            samples_per_scan=args.samples_per_scan, 
            load=args.loadData
        )

        val_loader = MMS3D(
            full_vol_dim=args.full_vol_dim, 
            root=args.root,
            scan_dir_path=args.scan_dir_path,
            labels_dir_path=args.labels_dir_path,
            mode='val', 
            crop_dim=args.dim, 
            split_id=args.split_id,
            samples_per_scan=args.samples_per_scan, 
            load=args.loadData
        )

    training_generator = DataLoader(train_loader, **params)
    val_generator = DataLoader(val_loader, **params)

    print("DATA SAMPLES HAVE BEEN GENERATED SUCCESSFULLY")
    return training_generator, val_generator



# Python libraries
import argparse
import os


# Lib files
import lib.medloaders as medical_loaders
import lib.medzoo as medzoo
import lib.train as train
import lib.utils as utils
from lib.losses3D import DiceLoss


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seed = 1777777



class args:
    loadData=True
    mode = 'train'
    split_id = (16,20,24)
    samples_per_scan = 2
    full_vol_dim = (640,141,385)
    dim = (640,32,32)
    root = '/media/przemek/c9923e93-365b-eda8-d61e-fbb64066d93e/MedicalZooPytorch'
    labels_dir_path = '/media/przemek/AEC095C3C09591E7/pdf/pp/10_Semestr/Praca magisterska/MMS2019/ref'
    scan_dir_path = '/media/przemek/AEC095C3C09591E7/pdf/pp/10_Semestr/Praca magisterska/MMS2019/skany3d'

    dataset_name = 'MMS3D'

    batchSz = 1
    shuffle = False
    num_workers = 1
    split = None
    
    samples_train = samples_per_scan*split_id[0]
    samples_val = samples_per_scan*(split_id[1]-split_id[0])

    split_last_dim = True

    inModalities = 1
    inChannels = 1
    cuda = False

    model = "UNET3D"
    opt = "sgd"
    lr = 5e-3
    classes = 1
    
    save = '../saved_models/' + model + '_checkpoints/' + model + '_{}_{}_'.format(
        utils.datestr(), dataset_name)

    log_dir = '../runs/'
    classes = 1
    terminal_show_freq = 2
    nEpochs = 2


from lib.losses3D import OCT2DBinaryCrossEntropyWithLogitsLoss



def main():
    # args = get_arguments()

    utils.reproducibility(args, seed)
    utils.make_dirs(args.save)

    # training_generator, val_generator, full_volume, affine = medical_loaders.generate_datasets(args,
    #                                                                                            path='.././datasets')
    training_generator, val_generator = generate_datasets(args)


    model, optimizer = medzoo.create_model(args)
    # criterion = DiceLoss(classes=args.classes)
    criterion = OCT2DBinaryCrossEntropyWithLogitsLoss()

    if args.cuda:
        model = model.cuda()
        print("Model transferred in GPU.....")

    trainer = train.Trainer(args, model, criterion, optimizer, train_data_loader=training_generator,
                            valid_data_loader=val_generator, lr_scheduler=None)
    print("START TRAINING...")
    trainer.training()



main()
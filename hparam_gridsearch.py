class hparams:
    output_dir = 'logs/' 
    aug = True 
    total_epochs = 600
    epochs_per_checkpoint = 10
    num_worker = 12
    batch_size = 20
    init_lr = 8e-5 #8e-5 #0.0002
    scheduer_step_size = 50
    scheduer_gamma = 0.9			
    latest_checkpoint_file = 'checkpoint_0800.pt'
    ckpt = None
    
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 2

    crop_or_pad_size = 200,200,40 #  WHD

    fold_arch = '*.nii.gz'

    source_data_dir = r'/home/dataset/ds_f'




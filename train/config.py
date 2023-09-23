settings = dict(
    lr_D = 4e-4, # learning rate of discriminator
    lr_G = 1e-4, # learning rate of generator
    target_len = 3000, # length of learned waveforms
    epoch = 10000, # number of training epoch
    seed = 0, # seed number for benchmark
    batch_size = 64, # batch size for training
    num_workers = 0, # how many subprocesses to use for data loading. 
                     # 0 means that the data will be loaded in the main process.
    );
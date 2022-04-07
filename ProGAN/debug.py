import config
step = 8
for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
    print(num_epochs)
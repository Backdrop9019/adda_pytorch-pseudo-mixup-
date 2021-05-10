

# params for dataset and data loader
data_root = "data"
batch_size = 128
image_size = 32

#for window
root = "C:\\Users\\USER\\Desktop\\캡디\\ADDA\\adda_pytorch\\"

#restore
use_restore = True

## sorce target
src_dataset = 'usps'

tgt_dataset = 'MNIST'

# params for target dataset
# 'mnist_m', 'usps', 'svhn'

##
src_encoder_restore = 'generated\\models\\'+src_dataset+'2'+tgt_dataset+'\\ADDA-source-encoder-final.pt'
src_classifier_restore = 'generated\\models\\'+src_dataset+'2'+tgt_dataset+'\\ADDA-source-classifier-final.pt'
src_model_trained = True


# target
tgt_encoder_restore = 'generated\\models\\'+src_dataset+'2'+tgt_dataset+'\\ADDA-target-encoder-final.pt'
tgt_model_trained = False




#dataset root
mnist_dataset_root = data_root
mnist_m_dataset_root = data_root+'\\mnist_m'
usps_dataset_root = data_root+'\\usps'
svhn_dataset_root = data_root+'\\svhn'
custom_dataset_root = data_root+'\\customdata\\train'


# params for training network
num_gpu = 1
num_epochs_pre = 60
num_epochs = 60
log_step_pre = 60
log_step = 60
eval_step_pre = 1 
##epoch
save_step_pre = 100
save_step = 100
manual_seed = 9590

d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = 'generated\\models\\'+src_dataset+'2'+tgt_dataset+'\\ADDA-critic-final.pt'

# params for optimizing models
c_learning_rate = 1e-4
d_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

model_root = 'generated\\models'

## mixup
usemixup = False
lammax = 0.2
lammin = 0.2

#labelsmoothing
labelsmoothing = False
smoothing = 0.2
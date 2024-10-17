from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--out_wt', type=float, default=0.75, help='Weight for output vs. deep layer outputs')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--optimizer', type=str, default='Adam', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--loss', type=str, default='ce', help='Loss to use? Default is combined dice and cross-entropy loss: choices are: dice, tversky, focal, soft_dsc')
        self.parser.add_argument('--model_type', type=str, default='deep', help='model type to use? Default is standard, other options are ''deep'' for deep supervision from an extra layer; ''multi'' for deep supervised classification; ''classifier'' for classification only')
        self.parser.add_argument('--model_name', type=str, default='MRRNDS_model', help='model name to test.')
        self.parser.add_argument('--nslices', type=int, default=5, help='slices for modality type')
        self.parser.add_argument('--use_aug', type=int, default=0,help='do *not* use augmentation on training data')  
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--T_mult', type=int, default=2, help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--T_0', type=int, default=5000, help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')

        self.parser.add_argument('--optimizer', type=str, default='AdamW', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--loss', type=str, default='dice', help='Loss to use? Default is combined dice and cross-entropy loss: choices are: dice, tversky, focal, soft_dsc')
        self.parser.add_argument('--isTrain', type=int, default=True, help='Loss to use? Default is combined dice and cross-entropy loss: choices are: dice, tversky, focal, soft_dsc')


        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        
        
        
        self.parser.add_argument('--mixup_betadist',  type=float, default=1.0, help='values of alpha and beta to define beta distribtiion for mixup lambda')
        self.parser.add_argument('--unfreeze_fraction1',  type=float, default=0.1, help='values of alpha and beta to define beta distribtiion for mixup lambda')
        self.parser.add_argument('--unfreeze_fraction2',  type=float, default=0.5, help='values of alpha and beta to define beta distribtiion for mixup lambda')
        self.parser.add_argument('--extra_neg_slices', type=int, default=1, help='added slices on either side of lesion for negative examples')

        # self.parser.add_argument('--unfreeze_fraction3',  type=float, default=1.0, help='values of alpha and beta to define beta distribtiion for mixup lambda')
        # self.parser.add_argument('--unfreeze_fraction4',  type=float, default=1.0, help='values of alpha and beta to define beta distribtiion for mixup lambda')
        
        
        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of computing vlidaiton results on screen')
        #self.parser.add_argument('--ct_seg_val_freq', type=int, default=2000, help='frequency of showing training results on screen')
        #self.parser.add_argument('--CT_merge_alpha', type=float, default=0.25, help='frequency of showing training results on screen')
        #self.parser.add_argument('--out_wt', type=float, default=0.75, help='Weight for output vs. deep layer outputs')
        #self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        #self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        #self.parser.add_argument('--use_ensembling', type=int, default=0, help='use ensembling to model uncertainty')
        #self.parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
        #self.parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        #self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        #self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        #self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        #model.to(device)self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--optimizer', type=str, default='AdamW', help='which epoch to load? set to latest to use latest cached model')
        #self.parser.add_argument('--loss', type=str, default='dice', help='Loss to use? Default is combined dice and cross-entropy loss: choices are: dice, tversky, focal, soft_dsc')
        #self.parser.add_argument('--model_type', type=str, default='deep', help='model type to use? Default is standard, other options are ''deep'' for deep supervision from an extra layer; ''multi'' for deep supervised classification; ''classifier'' for classification only')
        #self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        #self.parser.add_argument('--batchsize_adj_lr', type=int, default=0, help='# adjust initial lr by lr=lr*batch_size')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--stn_lr', type=float, default=0.00002, help='initial learning rate for adam')
        #self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        #self.parser.add_argument('--use_aug', type=int, default=0,help='do *not* use augmentation on training data')  
        #self.parser.add_argument('--use_mixup', type=int, default=0,help='do *not* use augmentation on training data')   
        #self.parser.add_argument('--use_label_smoothing', type=int, default=0,help='do *not* use augmentation on training data')   
        #self.parser.add_argument('--smoothing_alpha', type=float, default=0.01,help='do *not* use augmentation on training data')   
        #self.parser.add_argument('--online_label_softening', type=int, default=0,help='do *not* use augmentation on training data')   
        #self.parser.add_argument('--mm_layers',type=int,nargs="+", default=None,help='layers of MRRN to perform manifold mixup on')   
        #self.parser.add_argument('--mm_alpha', type=float, default=0.2,help='value for alpha in beta distribtion for manifold mixup interpolation')   
        #self.parser.add_argument('--mmixup_threshold', type=float, default=0.0,help='probability mixup will be implemented for layers in mm_layers')   
        self.parser.add_argument('--DSC_smoothing', type=float, default=0.0001,help='size of smoothing parameter in soft DSC calculation')   
        #self.parser.add_argument('--choose_mm_layer', type=int, default=1,help='choose one layer for mixup each time? Y/N')   
        #self.parser.add_argument('--ValidationType', type=str, default='2D', help='slicewise DSC or 3D volumetric DSC...')
        #self.parser.add_argument('--nshot', type=int, default=3,help='select how many patients to include in few shot learning (only applies in that case)')   
        #self.parser.add_argument('--incoherent_mixup', type=int, default=0,help='Test option for incoherent version of mixup. you probably want this as 0.')   
        #self.parser.add_argument('--noisyfm', action='store_true', default=False, help='Implement noisy version of feature mixup (must be using model with mixup enabled)')   
        #
        self.parser.add_argument('--model_to_test', type=str, default='AVG_best',help='key name for model to use AVG_best is default for early stopping')   
        #self.parser.add_argument('--dont_save', action='store_true', default=False,help='add to prevent code from saving tested volumes')   
        #self.parser.add_argument('--trainset', type=str, default='fast',help='which setting fro training data (default, fast is just positive slices, "extra" adds slices around lesion')   
 
        #self.parser.add_argument('--medisum', type=int, default=0,help='check whether only validated on Medisum tumor')    
        #self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        #self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        #self.parser.add_argument('--SegLambda_A',type=float,default=5,help='seg_weight for A')
        #self.parser.add_argument('--StyleLambda_A',type=float,default=2,help='style_weight for A')
        #self.parser.add_argument('--StyleLambda_B',type=float,default=2,help='style_weight for B')
        #self.parser.add_argument('--SegLambda_B',type=float,default=5,help='seg_weight for B')
        #self.parser.add_argument('--lambda_GA',type=float,default=1,help='GA for GA')       
        #self.parser.add_argument('--FeatureLambda',type=float,default=5,help='feature lambda to keep feature of two streams')        
        #self.parser.add_argument('--Load_CT_Weight_Seg_A',type=str,default='/lila/data/deasy/data_harini/headneck/checkpoints_unet/seg_CT/latest',help='pretrained_seg_parameters for Seg_A')                
        #self.parser.add_argument('--Load_CT_Weight_Seg_A',type=str,default='/lila/data/deasy/Josiah_data/pCR/checkpoints_unet/seg_CT/latest',help='pretrained_seg_parameters for Seg_A')                
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        #self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        #self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        #self.parser.add_argument('--T_mult', type=int, default=2, help='learning rate policy: lambda|step|plateau')
        #self.parser.add_argument('--T_0', type=int, default=5000, help='learning rate policy: lambda|step|plateau')
        
        
        
        # self.parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
        # self.parser.add_argument('--mri_D', type=int, default=1, help='use D to filter MRI')
        # self.parser.add_argument('--use_D_positive', type=int, default=0, help='use D to filter MRI')
        # self.parser.add_argument('--freez_G', type=int, default=0, help='use D to filter MRI')
        # self.parser.add_argument('--use_D_minimus', type=int, default=0, help='use D to filter MRI')
        # self.parser.add_argument('--lambda_feature', type=float, default=0.1, help='feature constraints weight on the two model')
        # self.parser.add_argument('--MMD_FeatureLambda', type=float, default=0.1, help='MMD_feature constraints weight on the two model')      
        # self.parser.add_argument('--use_feature_loss', type=int, default=1, help='whether use feature loss')
        # self.parser.add_argument('--sb_batch_per_gpu', type=int, default=1, help='batches per gpu when multi-gpu used')        
        # self.parser.add_argument('--use_style_loss', type=int, default=0, help='whether use feature loss')
        # self.parser.add_argument('--use_MMD_feature', type=int, default=0, help='whether use feature loss')    
        
        
        #self.parser.add_argument('--seg_theshold', type=float, default=0.5, help='threshold for seg')
        
        #self.parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        #self.parser.add_argument('--model_name', type=str, default='AVG_best', help='model name to test.')
        #self.parser.add_argument('--nslices', type=int, default=5, help='slices for modality type')
        #self.parser.add_argument('--rater', type=int, default=1, help='rater for interrater variablilty (1 or 2)')
        #self.parser.add_argument('--study', type=str, default='BL', help='Study in longitudinal data to investigate, BL, 3MO, 6MO, etc...')
        #self.parser.add_argument('--soft_mask', type=str, default='none', help='Superpixel softened mask for training, eg. slic2d, slic3d, etc')
        
        
        self.isTrain = True
        #self.parser.add_argument('--model', type=str, default='circle_gan_unet', help='multiply by a gamma every lr_decay_iters iterations')
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of displaying average loss')
		self.parser.add_argument('--niter', type=int, default=300, help='# of epochs to train')
		self.parser.add_argument('--learning_rate_decrease_itr', type=int, default=-1, help='how often is the learning rate decreased by six percent')
		self.parser.add_argument('--decay_factor', type=float, default=0.94, help='learning rate decay factor')
		self.parser.add_argument('--validation_on', action='store_true', help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=400, help='frequency of testing on validation set')
		self.parser.add_argument('--epoch_save_freq', type=int, default=5, help='frequency of saving intermediate models')

		#model arguments
		self.parser.add_argument('--init_material_weight', type=str, default= '', help='path to the pre-trained material net')
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=3, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")

		#optimizer arguments
		self.parser.add_argument('--lr_visual', type=float, default=0.0001, help='learning rate for visual stream')
		self.parser.add_argument('--lr_audio', type=float, default=0.0001, help='learning rate for audio')
		self.parser.add_argument('--lr_attention', type=float, default=0.0001, help='learning rate for attention network')
		self.parser.add_argument('--lr_material', type=float, default=0.0001, help='learning rate for material network')
		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0005, type=float, help='weights regularizer')

		self.mode = "train"
		self.isTrain = True
		self.enable_data_augmentation = True
		self.enable_cropping = False
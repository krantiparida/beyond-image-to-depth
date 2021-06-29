import os
import time
import torch
from options.train_options import TrainOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data_loader.custom_dataset_data_loader import CustomDatasetDataLoader
from util.util import TextWrite, compute_errors
import numpy as np
from models import criterion 

def create_optimizer(nets, opt):
	(net_visualdepth, net_audiodepth, net_attention, net_material) = nets
	param_groups = [{'params': net_rgbdepth.parameters(), 'lr': opt.lr_visual},
                    {'params': net_audiodepth.parameters(), 'lr': opt.lr_audio},
                    {'params': net_attention.parameters(), 'lr': opt.lr_attention},
                    {'params': net_material.parameters(), 'lr': opt.lr_material}
                    ]
	if opt.optimizer == 'sgd':
		return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
	elif opt.optimizer == 'adam':
		return torch.optim.Adam(param_groups, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)

def decrease_learning_rate(optimizer, decay_factor=0.94):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor

def evaluate(model, loss_criterion, dataset_val, opt):
	losses = []
	errors = []
	with torch.no_grad():
		for i, val_data in enumerate(dataset_val):
			output = model.forward(val_data)
			depth_predicted = output['depth_predicted']
			depth_gt = output['depth_gt']
			loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
			losses.append(loss.item())
			for idx in range(depth_predicted.shape[0]):
				errors.append(compute_errors(depth_gt[idx].cpu().numpy(), 
								depth_predicted[idx].cpu().numpy()))
	
	mean_loss = sum(losses)/len(losses)
	mean_errors = np.array(errors).mean(0)	
	print('Loss: {:.3f}, RMSE: {:.3f}'.format(mean_loss, mean_errors[1])) 
	val_errors = {}
	val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
	val_errors['DELTA1'] = mean_errors[2] 
	val_errors['DELTA2'] = mean_errors[3]
	val_errors['DELTA3'] = mean_errors[4]
	return mean_loss, val_errors 

loss_criterion = criterion.LogDepthLoss()
opt = TrainOptions().parse()
opt.device = torch.device("cuda")

#### Log the results
loss_list = ['step', 'loss']
err_list = ['step', 'RMSE', 'ABS_REL', 'DELTA1', 'DELTA2', 'DELTA3']

train_loss_file = TextWrite(os.path.join(opt.expr_dir, 'train_loss.csv'))
train_loss_file.add_line_csv(loss_list)
train_loss_file.write_line()

val_loss_file = TextWrite(os.path.join(opt.expr_dir, 'val_loss.csv'))
val_loss_file.add_line_csv(loss_list)
val_loss_file.write_line()

val_error_file = TextWrite(os.path.join(opt.expr_dir, 'val_error.csv'))
val_error_file.add_line_csv(err_list)
val_error_file.write_line()
################

# network builders
builder = ModelBuilder()
net_audiodepth = builder.build_audiodepth()
net_rgbdepth = builder.build_rgbdepth()
net_attention = builder.build_attention()
net_material = builder.build_material_property(init_weights=opt.init_material_weight)
# exit()
nets = (net_rgbdepth, net_audiodepth, net_attention, net_material)

# construct our audio-visual model
model = AudioVisualModel(nets, opt)
model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
model.to(opt.device)

dataloader = CustomDatasetDataLoader()
dataloader.initialize(opt)
dataset = dataloader.load_data()
dataset_size = len(dataloader)
print('#training clips = %d' % dataset_size)

if opt.validation_on:
	opt.mode = 'val'
	dataloader_val = CustomDatasetDataLoader()
	dataloader_val.initialize(opt)
	dataset_val = dataloader_val.load_data()
	dataset_size_val = len(dataloader_val)
	print('#validation clips = %d' % dataset_size_val)
	opt.mode = 'train'

optimizer = create_optimizer(nets, opt)

# initialization
total_steps = 0
batch_loss = []
best_rmse = float("inf")
best_loss = float("inf")

for epoch in range(1, opt.niter+1):
	torch.cuda.synchronize()
	batch_loss = []   

	for i, data in enumerate(dataset):

		total_steps += opt.batchSize
		
		# forward pass
		model.zero_grad()
		output = model.forward(data)
		
		# calculate loss
		depth_predicted = output['depth_predicted']
		depth_gt = output['depth_gt']
		loss = loss_criterion(depth_predicted[depth_gt!=0], depth_gt[depth_gt!=0])
		batch_loss.append(loss.item())

		# update optimizer
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if(total_steps // opt.batchSize % opt.display_freq == 0):
			print('Display training progress at (epoch %d, steps %d)' % (epoch, total_steps // opt.batchSize))
			avg_loss = sum(batch_loss) / len(batch_loss)
			print('Average loss: %.5f' % (avg_loss))
			batch_loss = []
			print('end of display \n')
			train_loss_file.add_line_csv([total_steps // opt.batchSize, avg_loss])
			train_loss_file.write_line()
			
		if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
			model.eval()
			opt.mode = 'val'
			print('Display validation results at (epoch %d, steps %d)' % (epoch, total_steps // opt.batchSize))
			val_loss, val_err = evaluate(model, loss_criterion, dataset_val, opt)
			print('end of display \n')
			model.train()
			opt.mode = 'train'

			# save the model that achieves the smallest validation error
			if val_err['RMSE'] < best_rmse:
				best_rmse = val_err['RMSE']
				print('saving the best model (epoch %d) with validation RMSE %.5f\n' % (epoch, val_err['RMSE']))
				torch.save(net_rgbdepth.state_dict(), os.path.join(opt.expr_dir, 'rgbdepth_'+opt.dataset+'.pth'))
				torch.save(net_audiodepth.state_dict(), os.path.join(opt.expr_dir, 'audiodepth_'+opt.dataset+'.pth'))
				torch.save(net_attention.state_dict(), os.path.join(opt.expr_dir, 'attention_'+opt.dataset+'.pth'))
				torch.save(net_material.state_dict(), os.path.join(opt.expr_dir, 'material_'+opt.dataset+'.pth'))

			
			#### Logging the values for the val set
			val_loss_file.add_line_csv([total_steps // opt.batchSize, val_loss])
			val_loss_file.write_line()
			
			err_list = [total_steps // opt.batchSize, \
				val_err['RMSE'], val_err['ABS_REL'], \
				val_err['DELTA1'], val_err['DELTA2'], val_err['DELTA3']]
			val_error_file.add_line_csv(err_list)
			val_error_file.write_line()

	if epoch % opt.epoch_save_freq == 0:
		print('saving the model at 5th epoch')
		torch.save(net_rgbdepth.state_dict(), os.path.join(opt.expr_dir, 'rgbdepth_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
		torch.save(net_audiodepth.state_dict(), os.path.join(opt.expr_dir, 'audiodepth_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
		torch.save(net_attention.state_dict(), os.path.join(opt.expr_dir, 'attention_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))
		torch.save(net_material.state_dict(), os.path.join(opt.expr_dir, 'material_'+opt.dataset+'_epoch_'+str(epoch)+'.pth'))

	#decrease learning rate 6% every opt.learning_rate_decrease_itr epochs
	if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
		decrease_learning_rate(optimizer, opt.decay_factor)
		print('decreased learning rate by ', opt.decay_factor)

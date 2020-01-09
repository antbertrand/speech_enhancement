import os
import torch
import matplotlib.pyplot as plt 

def main():
	folder = 'experiment1/fs8000_snr1_nfft256_hop128'
	chkpt_name = os.listdir(folder)[-1]
	chkpt_path = os.path.join(folder, chkpt_name)
	chkpt = torch.load(chkpt_path)

	logs = chkpt['logs']

	epoch = logs['_TrainingHistory__epoch']
	train_loss = logs['_TrainingHistory__train_loss']
	val_loss = logs['_TrainingHistory__val_loss']

	print(epoch)

	plt.figure()
	plt.plot(train_loss, '-b', label='train loss')
	plt.plot(val_loss, '-g', label='val loss')
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main()
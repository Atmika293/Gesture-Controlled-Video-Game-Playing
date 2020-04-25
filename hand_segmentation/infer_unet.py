import numpy as np
import torch
import cv2
import os

from torchvision import transforms

import argparse

def predict(net, image, normalize, gpu):
	image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_NEAREST)
	image = torch.unsqueeze(normalize(image), dim=0)

	if gpu:
		image = image.cuda()
	
	mask = torch.round(net(image))
	output = torch.mul(image, mask)

	if gpu:
		mask = mask.cpu()
		output = output.cpu()

	final_image = output.detach().numpy().astype('uint8').squeeze().transpose(1, 2, 0)
	final_mask = mask.detach().numpy().astype('uint8').squeeze() * 255
	
	return final_image, final_mask

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='network.pth', help='model path')
	parser.add_argument('--outf', type=str, default='./test_results', help='output folder')
	parser.add_argument('--dataset', type=str, required=True, help="dataset path")
	parser.add_argument('--gpu', action='store_true', help="use feature transform")
	opt = parser.parse_args()

	if not opt.gpu:
		net = torch.load(opt.model, map_location=torch.device('cpu'))
	else:
		net = torch.load(opt.model).cuda()

	normalize = transforms.ToTensor()
	net.eval()

	if not os.path.isdir(opt.outf):
		os.mkdir(opt.outf)

	for filename in os.listdir(opt.dataset):
		print(filename)
		image = cv2.imread(os.path.join(opt.dataset, filename))
		output, mask = predict(net, image, normalize, opt.gpu)

		cv2.imwrite(os.path.join(opt.outf, filename), output)
		cv2.imwrite(os.path.join(opt.outf, 'mask_' + filename), mask)
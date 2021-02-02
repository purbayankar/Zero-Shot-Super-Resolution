import torch
import torch.nn as nn
import math
from eca_module import eca_layer


# use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')

class SRNet(nn.Module):

	def __init__(self):
		super(SRNet, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		self.pad = nn.ReflectionPad2d(1)

		self.Conv1 = nn.Conv2d(3,64,3,1,1,bias=True)
		self.Conv1_ = nn.Conv2d(3,64,5,1,1,bias=True)
		self.Conv2 = nn.Conv2d(128,64,3,1,1,bias=True)
		self.Conv2_ = nn.Conv2d(128,64,5,1,1,bias=True)
		self.Conv3 = nn.Conv2d(128,64,3,1,1,bias=True)
		self.Conv4 = nn.Conv2d(64,3,3,1,1,bias=True)
		self.Conv5 = nn.Conv2d(3,64,3,1,1,bias=True)
		self.Conv5_ = nn.Conv2d(3,64,5,1,1,bias=True)
		self.Conv6 = nn.Conv2d(128,64,3,1,1,bias=True)
		self.Conv6_ = nn.Conv2d(128,64,5,1,1,bias=True)
		self.Conv7 = nn.Conv2d(128,64,3,1,1,bias=True)
		self.Conv8 = nn.Conv2d(64,3,3,1,1,bias=True)
		self.eca = eca_layer(3,3)


	def forward(self, LR_img):
		
		x = self.relu(self.Conv1(LR_img))
		x_ = self.relu(self.Conv1_(self.pad(LR_img)))
		x = torch.cat((x,x_), dim=1)
		x1 = self.relu(self.Conv2(x))
		x1_ = self.relu(self.Conv2_(self.pad(x)))
		x2 = torch.cat((x1,x1_), dim=1)
		x2 = self.relu(self.Conv3(x2))
		x2 = self.Conv4(x2)
		x2 = self.eca(x2)
		x2 = x2 + LR_img
		x3 = self.relu(self.Conv5(x2))
		x3_ = self.relu(self.Conv5_(self.pad(x2)))
		x4 = torch.cat((x3,x3_), dim=1)
		x5 = self.relu(self.Conv6(x4))
		x5_ = self.relu(self.Conv6_(self.pad(x4)))
		x6 = torch.cat((x5,x5_), dim=1)
		x6 = self.relu(self.Conv7(x6))
		x6 = self.Conv8(x6)
		x6 = self.eca(x6)
		
		x6 = x6 + x2
		
# 		SR_img = LR_img + x6     # Because we have to learn residuals.

		return x6

# class SRNet(nn.Module):

# 	def __init__(self):
# 		super(SRNet, self).__init__()

# 		self.relu = nn.ReLU(inplace=True)

# 		self.Conv1 = nn.Conv2d(3,64,3,1,1,bias=True)
# 		self.Conv2 = nn.Conv2d(64,64,3,1,1,bias=True)
# 		self.Conv3 = nn.Conv2d(64,64,3,1,1,bias=True)
# 		self.Conv4 = nn.Conv2d(64,64,3,1,1,bias=True)
# 		self.Conv5 = nn.Conv2d(64,64,3,1,1,bias=True)
# 		self.Conv6 = nn.Conv2d(64,64,3,1,1,bias=True)
# 		self.Conv7 = nn.Conv2d(64,64,3,1,1,bias=True)
# 		self.Conv8 = nn.Conv2d(64,3,3,1,1,bias=True)

# 	def forward(self, LR_img):

# 		x = self.relu(self.Conv1(LR_img))
# 		x = self.relu(self.Conv2(x))
# 		x = self.relu(self.Conv3(x))
# 		x = self.relu(self.Conv4(x))
# 		x = self.relu(self.Conv5(x))
# 		x = self.relu(self.Conv6(x))
# 		x = self.relu(self.Conv7(x))
# 		x = self.Conv8(x)

# 		SR_img = LR_img + x     # Because we have to learn residuals.

# 		return SR_img


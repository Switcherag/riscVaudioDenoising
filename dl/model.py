import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, chnls_in=1, chnls_out=1):
        super(UNetGenerator, self).__init__()
        self.down_conv_layer_1 = DownConvBlock(chnls_in, 64, norm=False)
        self.down_conv_layer_2 = DownConvBlock(64, 128)
        self.down_conv_layer_3 = DownConvBlock(128, 256)
        self.down_conv_layer_4 = DownConvBlock(256, 256, dropout=0.5)
        self.down_conv_layer_5 = DownConvBlock(256, 256, dropout=0.5)
        self.down_conv_layer_6 = DownConvBlock(256, 256, dropout=0.5)

        self.up_conv_layer_1 = UpConvBlock(256, 256, kernel_size=(2,3), stride=2, padding=0, dropout=0.5)# 256+256 6 5 kernel_size=(2, 3), stride=2, padding=0
        self.up_conv_layer_2 = UpConvBlock(512, 256, kernel_size=(2,3), stride=2, padding=0, dropout=0.5) # 256+256 1 4
        self.up_conv_layer_3 = UpConvBlock(512, 256, kernel_size=(2,3), stride=2, padding=0, dropout=0.5) # 2 3
        self.up_conv_layer_4 = UpConvBlock(512, 128, dropout=0.5) # 3 2
        self.up_conv_layer_5 = UpConvBlock(256, 64) # 4 1
        self.up_conv_layer_6 = UpConvBlock(512, 128)
        self.up_conv_layer_7 = UpConvBlock(256, 64)
        self.upsample_layer = nn.Upsample(scale_factor=2)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        self.conv_layer_1 = nn.Conv2d(128, chnls_out, 4, padding=1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        #print('x', x.shape)
        enc1 = self.down_conv_layer_1(x) # [4, 64, 32, 188]
        #print('1', enc1.shape)
        enc2 = self.down_conv_layer_2(enc1) # [4, 128, 16, 94]
        #print('2', enc2.shape)
        enc3 = self.down_conv_layer_3(enc2) # [4, 256, 8, 47]
        #print('3', enc3.shape)
        enc4 = self.down_conv_layer_4(enc3) # [4, 256, 4, 23]
        #print('4', enc4.shape)
        enc5 = self.down_conv_layer_5(enc4) # [4, 256, 2, 11]
        #print('5', enc5.shape)
        enc6 = self.down_conv_layer_6(enc5) # [4, 256, 1, 5]
        #print('6', enc6.shape)
 
        dec1 = self.up_conv_layer_1(enc6, enc5)# enc6: 256 + enc5: 256 [4, 512, 2, 11]
        #print('d1', dec1.shape)
        dec2 = self.up_conv_layer_2(dec1, enc4)# enc4: 256 + dec1=enc5*2: [4, 512, 4, 23]
        #print('d2', dec2.shape)
        dec3 = self.up_conv_layer_3(dec2, enc3)# enc3: 256 + dec2=enc4*2: [4, 512, 8, 47]
        #print('d3', dec3.shape)
        dec4 = self.up_conv_layer_4(dec3, enc2)# enc2: 128 + dec3=enc3*2: [4, 256, 16, 94]
        #print('d4', dec4.shape)
        dec5 = self.up_conv_layer_5(dec4, enc1)# enc1: 64 + dec4=enc1*2: [4, 128, 32, 188]
        #print('d5', dec5.shape)
      
        final = self.upsample_layer(dec5)
        final = self.zero_pad(final)
        final = self.conv_layer_1(final)
        #print(final.shape)
        return final

class UpConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, stride= 2, padding=1 ,dropout=0.0):
        super(UpConvBlock, self).__init__()
        self.layers = [
            nn.ConvTranspose2d(ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(op_sz),
            nn.ReLU(),
        ]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
    def forward(self, x, enc_ip):
        x = nn.Sequential(*(self.layers))(x)
        #print('x', x.shape)
        #print('enc', enc_ip.shape)
        op = torch.cat((x, enc_ip), 1)
        return op


class DownConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, norm=True, dropout=0.0):
        super(DownConvBlock, self).__init__()
        self.layers = [nn.Conv2d(ip_sz, op_sz, kernel_size, 2, 1)]
        if norm:
            self.layers.append(nn.InstanceNorm2d(op_sz))
        self.layers += [nn.LeakyReLU(0.2)]
        if dropout:
            self.layers += [nn.Dropout(dropout)]
    def forward(self, x):
        op = nn.Sequential(*(self.layers))(x)
        return op

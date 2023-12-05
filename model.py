
import torch
import torch.nn as nn
import yolo_components as yolo


class VetHead(nn.Module):
    def __init__(self):
        super(VetHead, self).__init__()



class VetNet(nn.Module):
    def __init__(self):
        super(VetNet, self).__init__()
        self.Conv_1 = yolo.Conv(3,64,6,2,2)
        self.Conv_2 = yolo.Conv(64,128, 3, 2, 1)
        self.C3_1 = yolo.C3(128,128, 1)
        self.Conv_3 = yolo.Conv(128, 256, 3, 2, 1)
        self.C3_2 = yolo.C3(256, 256, 2)
        self.Conv_4 = yolo.Conv(256, 512, 3, 2, 1)
        self.C3_3 = yolo.C3(512, 512, 3)
        #self.Conv_5 = yolo.Conv(512, 1024, 3, 2, 1)
        #self.C3_4 = yolo.C3(1024, 1024, 1)
        #self.SPFF = yolo.SPPF(1024, 1024, 5)
        #End of Backbone

        #self.Conv_6 = yolo.Conv(1024, 512, 1, 1, 0)
        #self.upSample = nn.Upsample(None, 2, 'nearest')
        #self.concat = yolo.Concat()
        #self.C3_5 = yolo.C3(1024, 512, False)
        #self.Conv_7 = yolo.Conv(512, 256, 1, 1, 0)
        #upsample and concat
        #self.C3_6 = yolo.C3(512, 256, 1, False)
        #concat
        #self.Conv_8 = yolo.Conv(256, 256, 3, 2, 1)
        #concat
        #self.C3_7 = yolo.C3(512, 512, 1, False)
        #self.Conv_9 = yolo.Conv(512, 512, 3, 2, 1)
        #concat
        #self.C3_8 = yolo.C3(1024, 1024, 1, False)

        #self.Conv_a = nn.Conv2d(256, 4, 1, 1, 0)
        #self.Conv_b = nn.Conv2d(512, 4, 1, 1, 0)
        #self.Conv_c = nn.Conv2d(1024, 4, 1, 1, 0)

        #self.avgPool_a = nn.AdaptiveAvgPool2d(100)
        #self.avgPool_b = nn.AdaptiveAvgPool2d(100)
        #self.avgPool_c = nn.AdaptiveAvgPool2d(100)

        self.fc_out = nn.Linear(512 * 40 * 40, 2)
        self.ReLu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        #self.Conv_Y_1 = yolo.Conv(512, 10, 1, 1, 0)
        #self.Conv_2_1 = yolo.Conv(512, 256, 1, 1, 0)
        #self.Conv_2_2 = yolo.Conv(1024, 512, 1, 1, 0)


    def forward(self, x):
        x = self.Conv_1(x)
        x = self.Conv_2(x)
        x = self.C3_1(x)
        x = self.Conv_3(x)
        r1 = self.C3_2(x)
        x = self.Conv_4(r1)
        r2 = self.C3_3(x)
        #x = self.Conv_5(r2)
        #x = self.C3_4(x)
        #x = self.SPFF(x)
        #r3 = self.Conv_6(x)
        #x = self.upSample(r3)
        #x = torch.cat((x, r2), dim=1)
        #x = self.Conv_2_2(x)
        #r4 = self.Conv_7(x)
        #x = self.upSample(r4)
        #x = torch.cat((x, r1), dim=1)
        #ra = self.Conv_2_1(x)
        #x = self.Conv_8(ra)
        #rb = torch.cat((x, r4), dim=1)
        #rb = self.C3_7(x)
        #x = self.Conv_9(rb)
        #x = torch.cat((x, r3), dim=1)
        #rc = self.C3_8(x)

        #a = self.Conv_a(ra)
        #a = self.avgPool_a(a)

        #b = self.Conv_b(rb)
        #b = self.avgPool_b(b)

        #c = self.Conv_c(rc)
        #c = self.avgPool_c(c)

        #y = torch.cat((a, b, c), dim=1)
        #y = self.Conv_Y_1(r3)
        #print(y.shape)

        y = r2.view(r2.size(0), -1)  # Reshape y to [batch_size, 12 * 100 * 100]
        y = self.fc_out(y)
        print(y.shape)
        y = self.Sigmoid(y)

        return x

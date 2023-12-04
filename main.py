import model as net
from torchsummary import summary
from train import train
import os
if __name__ == '__main__':
    #os.remove("./data/images/Egyptian_Mau_167.jpg") #DONE
    #os.remove("./data/images/Egyptian_Mau_14.jpg") #DONE
    #os.remove("./data/images/Egyptian_Mau_139.jpg") #DONE (not in train/test)
    #os.remove("./data/images/Egyptian_Mau_145.jpg") #DONE (not in train/test)
    #os.remove("./data/images/Egyptian_Mau_156.jpg") #DONE
    #os.remove("./data/images/Egyptian_Mau_177.jpg") #DONE (not in train/test)
    #os.remove("./data/images/Egyptian_Mau_186.jpg") #DONE
    #os.remove("./data/images/Egyptian_Mau_191.jpg") #DONE (not in train/test)
    #os.remove("./data/images/Abyssinian_5.jpg") #DONE
    #os.remove("./data/images/Abyssinian_34.jpg") #DONE - preprocessing complete
    #os.remove("./data/images/chihuahua_121.jpg") #alaso corrupted image
    #os.remove("./data/image/beagle_116.jpg") #also corrupted image
    num_epochs = 1
    batch_size = 1
    model = net.VetNet()
    device = 'cuda'
    learning_rate = 0.005
    scaling_factor = 0.1
    patience = 3
    train(batch_size, num_epochs, model, device, learning_rate, scaling_factor, patience)



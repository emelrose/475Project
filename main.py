import model as net
from torchsummary import summary
from train import train
import os
if __name__ == '__main__':
    #os.remove("./data/images/Egyptian_Mau_167.jpg")
    #os.remove("./data/images/Egyptian_Mau_14.jpg")
    #os.remove("./data/images/Egyptian_Mau_139.jpg")
    #os.remove("./data/images/Egyptian_Mau_145.jpg")
    #os.remove("./data/images/Egyptian_Mau_156.jpg")
    #os.remove("./data/images/Egyptian_Mau_177.jpg")
    #os.remove("./data/images/Egyptian_Mau_186.jpg")
    #os.remove("./data/images/Egyptian_Mau_191.jpg")
    #os.remove("./data/images/Abyssinian_5.jpg")
    #os.remove("./data/images/Abyssinian_34.jpg")
    num_epochs = 20
    batch_size = 20
    model = net.VetNet()
    device = 'cuda'
    learning_rate = 0.001
    scaling_factor = 0.1
    patience = 3
    train(batch_size, num_epochs, model, device, learning_rate, scaling_factor, patience)



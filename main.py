import model as net
from torchsummary import summary
from train import train
if __name__ == '__main__':

    num_epochs = 1
    batch_size = 32
    model = net.VetNet()
    device = 'cpu'
    learning_rate = 0.001
    scaling_factor = 0.1
    patience = 3
    train(num_epochs, batch_size, model, device, learning_rate, scaling_factor, patience)


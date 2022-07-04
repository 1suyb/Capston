from Train.Train import Train
from Model.MPNCOV import MPNCOV
if __name__ == "__main__" :
    train = Train(dataset_path=r".\Data", n_epoch=25, model="SRGAN")
    train.Train_ND()
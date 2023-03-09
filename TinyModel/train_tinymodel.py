import torch
from torch.cuda.nvtx import range_push,range_pop

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        range_push("Linear 1")
        x = self.linear1(x)
        range_pop()
        range_push("ReLU")
        x = self.activation(x)
        range_pop()
        range_push("Linear 2")
        x = self.linear2(x)
        range_pop()
        range_push("Softmax")
        x = self.softmax(x)
        range_pop()
        return x

def main(sample_size = 10*2**10):
    tinymodel = TinyModel()
    print('The model:')
    print(tinymodel)

    trainset = torch.rand(10, sample_size)
    target = torch.rand(10)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(tinymodel.parameters(), lr=0.05)

    for i,sample in enumerate(trainset):
        range_push(f"MiniBatch {i}")
        output = tinymodel(sample)
        loss = criterion(output,target)
        print(f'Loss = {loss}')
        optimizer.zero_grad()
        range_push("Backpropagation")
        loss.backward()
        range_pop()
        range_push("Optimizer")
        optimizer.step()
        range_pop()
        range_pop()

if __name__ == '__main__':
    if torch.cuda.is_available():
        main()
    else:    
        print(f'No CUDA available. Bye!')
        
import torch
from torch.cuda.nvtx import range_push,range_pop

class TinyModel(torch.nn.Module):

    def __init__(self, input_size):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(input_size, 200)
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

def main(input_size = 10*2**10):
    tinymodel = TinyModel(input_size)
    print('The model:')
    print(tinymodel)
    tinymodel = tinymodel.to('cuda:0')

    trainset = torch.rand(10, input_size).to('cuda:0')
    target = torch.rand(10).to('cuda:0')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(tinymodel.parameters(), lr=0.05)

    for i,sample in enumerate(trainset):
        range_push(f"MiniBatch {i}")
        range_push(f"Forward")
        output = tinymodel(sample)
        range_pop()
        range_push(f"Conmpute Loss")
        loss = criterion(output,target)
        range_pop()
        #print(f'Loss = {loss}')
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
        main(input_size=10*2**20)
    else:    
        print(f'No CUDA available. Bye!')
        

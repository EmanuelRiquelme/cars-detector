import torch

def save_model(model,optimizer):
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, 'dog_breed.pth.tar')

def load_model(model,optimizer,file_name='dog_breed.pth.tar'):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def validation(data,model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    it = iter(data)
    acc = []
    model.eval()
    with torch.no_grad():
        for _ in range(len(data)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            pred = model(input)
            pred = torch.argmax(pred,-1)
            acc.append((((pred == target).nonzero()).size(0)/target.size(0)))
        model.train()
        return sum(acc)/len(acc)

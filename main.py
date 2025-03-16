#%%
import Dpsnmodel
import PSdata
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = PSdata.PSDataset('data/train', device)
test_set = PSdata.PSDataset('data/test', device)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

model = Dpsnmodel.Dpsn(20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss = torch.nn.MSELoss()
#100
#%%
Dpsnmodel.train(model, train_loader, optimizer, loss, device, 100)

test_loss = Dpsnmodel.test(model, test_loader, loss, device)
print(f"Test loss: {test_loss}")

#%%
#save
torch.save(model.state_dict(), 'model.pth')
#%%
#load
model = Dpsnmodel.Dpsn(20).to(device)
model.load_state_dict(torch.load('model.pth'))


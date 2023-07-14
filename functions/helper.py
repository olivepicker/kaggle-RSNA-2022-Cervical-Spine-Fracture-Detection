import torch
import gc

def save_model_state(name, model, epoch, scheduler,optimizer):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, f'{name}_{epoch}epoch.pth')
    
    
def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()
    
def save_model(name, model):
    torch.save(model.state_dict(), f'{name}.tph')
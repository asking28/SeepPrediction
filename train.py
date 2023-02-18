import torch
import torch.nn as nn
import torch.nn.functional as F
from read_data import *
from read_model import *
from torchsummary import summary
import segmentation_models_pytorch as smp


model_name = 'unetplusplus'
encoder_name = 'resnet34' #mit_b3
encoder_weights = 'imagenet'


model = define_model(model_name,encoder_name,encoder_weights )

dataloader_train,dataloader_val = create_loader(32)
tvf_loss = smp.losses.TverskyLoss('multilabel', log_loss=False, from_logits=True, smooth=0.0, ignore_index=None, eps=1e-07, alpha=0.5, beta=0.5, gamma=1.0)
def calc_metrics(tp, fp, fn, tn):
  # first compute statistics for true positives, false positives, false negative and
  # true negative "pixels"
  # tp, fp, fn, tn = smp.metrics.get_stats(output, y, mode='multilabel', threshold=0.5)

  # then compute metrics with required reduction (see metric docs)
  iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
  f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
  f2_score = smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction="micro")
  accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
  recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
  m = {'iou':iou_score,'f1_score':f1_score,'f2_score':f2_score,'accuracy':accuracy,'recall':recall}
  return m

def train(model, optimizer, scheduler, loss_fn, num_epochs=1, evaluate=2):
    
    #Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device =',device)
    model=model.to(device)

    # Instantiate Tensorboard SummaryWriter
    writer = SummaryWriter()    
    
    best_loss = float('inf')
    total_train_samples = 0
    total_test_samples = 0   
    lr = optimizer.param_groups[0]['lr']
    # Iterate over epochs
    for epoch in tq.tqdm(range(num_epochs)):

        # print learning rate if it was changed in this epoch
        # otherwize don't print it
        if lr != optimizer.param_groups[0]['lr']:
            lr = optimizer.param_groups[0]['lr']
            print('learning rate =', lr)
        
        #Train      
        model.train()
        torch.set_grad_enabled(True)
        tp, fp, fn, tn = None,None,None,None
        for imgs, msks in dataloader_train:
            
            imgs = imgs.to(device).float()   
            msks = msks.to(device)
            msks = msks.squeeze(1)

            
            batch_size = imgs.size(0)
            total_train_samples += batch_size

            optimizer.zero_grad()
            msks_pred = model(imgs) 
               
  
            loss = loss_fn(msks_pred, msks)
              
            writer.add_scalar('train_loss', loss, total_train_samples)
            
            loss.backward()            
            optimizer.step()
            tp_temp, fp_temp, fn_temp, tn_temp = smp.metrics.get_stats(msks_pred.to(device), msks.to(device), mode='multilabel', threshold=0.5)
            if tp==None:
              tp = tp_temp
              fp = fp_temp
              fn = fn_temp
              tn = tn_temp
            else:
              tp = torch.cat([tp,tp_temp])
              fp = torch.cat([fp,fp_temp])
              fn = torch.cat([fn,fn_temp])
              tn = torch.cat([tn,tn_temp])       
        
        d1 = calc_metrics(tp, fp, fn, tn)
        for key in d1:
          print(key,": ",d1[key])
        #Eval        
        if epoch % evaluate == 0: # evaluate results for epochs # 0 and all multiples of evaluate
            val_gt, val_pred = None, None
            model.eval()
            torch.set_grad_enabled(False)
                   
            epoch_loss = 0            
            tp, fp, fn, tn = None, None, None, None
            for imgs, msks in dataloader_val:
                
                imgs = imgs.to(device).float()
                msks = msks.to(device)
                msks = msks.squeeze(1)
                batch_size = imgs.size(0)
                total_test_samples+=batch_size

                msks_pred = model(imgs)
                        
                loss = loss_fn(msks_pred, msks)
                
                writer.add_scalar('val_loss', loss, total_test_samples)                 
                
                epoch_loss += loss
                tp_temp, fp_temp, fn_temp, tn_temp = smp.metrics.get_stats(msks_pred.to(device), msks.to(device), mode='multilabel', threshold=0.5)
                if tp==None:
                  tp = tp_temp
                  fp = fp_temp
                  fn = fn_temp
                  tn = tn_temp
                else:
                  tp = torch.cat([tp,tp_temp])
                  fp = torch.cat([fp,fp_temp])
                  fn = torch.cat([fn,fn_temp])
                  tn = torch.cat([tn,tn_temp])       
            d1 = calc_metrics(tp, fp, fn, tn)
            for key in d1:
              print('Validation ',key,": ",d1[key])
            #Save best model
            save = '' # will print if model saved
            if epoch_loss < best_loss:
                save = 'model saved' # will print if model saved
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'current_best_model_state_dict.pth')
                
            print('epoch:', epoch,
                  'loss:', round(epoch_loss.item(), 3),  
                  save)
       
        #Step for learning rate scheduler
        scheduler.step()
    
    # Close SummaryWriter
    writer.close()
    
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.33)
model=train(model=model, optimizer=optimizer, scheduler=exp_lr_scheduler, loss_fn=tvf_loss,  num_epochs=50, evaluate=2)




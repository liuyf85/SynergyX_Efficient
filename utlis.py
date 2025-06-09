import os
import os.path as osp
import random
import numpy as np
import torch
from mmcv.utils import collect_env as collect_base_env
from torch_geometric.loader import DataLoader
from dataset.My_inMemory_dataset import MyInMemoryDataset
from metrics import get_metrics


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None):
        """
        Initialize EarlyStopping object.

        Args:
            mode (str): 'higher' if a higher score is better, 'lower' if a lower score is better.
            patience (int): Number of epochs to wait for improvement before early stopping.
            filename (str): Name of the checkpoint file to save the model state.
            metric (str): Metric to monitor for early stopping. Can be 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', or 'mse'.
            n_fold (int): Fold number used for naming checkpoint file.
            folder (str): Folder path to save checkpoint file.
        """

        if filename is None:
            filename = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):

        model.load_state_dict(torch.load(self.filename))


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    return env_info


def load_dataloader(n_fold,args):
    work_dir = args.workdir
    data_root = osp.join(work_dir,'data')

    if args.celldataset == 1:
        celllines_data = osp.join(data_root,'0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root,'0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
        # celllines_data = osp.join(data_root,'0_cell_data/4079g/170_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
        # celllines_data = osp.join(data_root,'0_cell_data/11587g/170_cellGraphs_exp_mut_cn_eff_dep_met_11587_genes_norm.npy')
        # celllines_data = osp.join(data_root,'0_cell_data/6707g/170_cellGraphs_exp_mut_cn_eff_dep_met_6707_genes_norm.npy')
        # celllines_data = osp.join(data_root,'0_cell_data/7596g/170_cellGraphs_exp_mut_cn_eff_dep_met_7596_genes_norm.npy')
        # celllines_data = osp.join(data_root,'0_cell_data/8467g/170_cellGraphs_exp_mut_cn_eff_dep_met_8467_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root,'0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')

    
    drugs_data = osp.join(data_root,'1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    tr_data_items = osp.join(data_root,f'split/{n_fold}_fold_tr_items.npy')
    val_data_items = osp.join(data_root,f'split/{n_fold}_fold_val_items.npy')  
    test_data_items = osp.join(data_root,f'split/{n_fold}_fold_test_items.npy')
    
    tr_dataset = MyInMemoryDataset(data_root,tr_data_items,celllines_data,drugs_data,args=args)
    val_dataset = MyInMemoryDataset(data_root,val_data_items,celllines_data,drugs_data,args=args)
    test_dataset = MyInMemoryDataset(data_root,test_data_items,celllines_data,drugs_data,args=args)

    # tr_dataloader = DataLoader(tr_dataset[:3200], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    # val_dataloader = DataLoader(val_dataset[:3200], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    # test_dataloader = DataLoader(test_dataset[:3200], batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
    
    print(f'train data:{len(tr_dataloader)*args.batch_size}')  
    print(f'Valid data:{len(val_dataloader)*args.batch_size}')
    print(f'Test data:{len(test_dataloader)*args.batch_size}')
    
    return tr_dataloader, val_dataloader, test_dataloader



def load_infer_dataloader(args):

    
    data_root = osp.join(args.workdir,'data')
    if args.celldataset == 1:
        celllines_data = osp.join(data_root,'0_cell_data/18498g/985_cellGraphs_exp_mut_cn_18498_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root,'0_cell_data/4079g/985_cellGraphs_exp_mut_cn_eff_dep_met_4079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root,'0_cell_data/963g/985_cellGraphs_exp_mut_cn_963_genes_norm.npy')
    drugs_data = osp.join(data_root,'1_drug_data/drugSmile_drugSubEmbed_2644.npy')

    data_items = args.infer_path 
    infer_dataset = MyInMemoryDataset(data_root,data_items,celllines_data,drugs_data,args=args)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=4)

    infer_data_arr = np.load(data_items,allow_pickle=True)

    return infer_dataloader,infer_data_arr



def train(model,criterion,opt,dataloader,device,scaler,args=None):
    from torch.amp import autocast, GradScaler
    import time
    model.train()
    train_loss_sum = 0
    i = 0 
    lr_list = []
    
    # from torch.profiler import profile, record_function, ProfilerActivity

    # with profile(
    #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #     record_shapes=False,
    #     with_stack=False
    # ) as prof:
    #     with record_function("train_loop"):
    #         for data in dataloader:
    #             i += 1
    #             if i > 100:
    #                 break
    #             opt.zero_grad()
    #             x = data.to(device, non_blocking=False)
    #             y = data.y.unsqueeze(1).type(torch.float32).to(device, non_blocking=False)
    #             with autocast(device_type = device, dtype=torch.float16):
    #                 output, _, _ = model(x)
    #                 train_loss = criterion(output, y)
    #             train_loss_sum += train_loss.detach().item()
    #             scaler.scale(train_loss).backward()
    #             scaler.step(opt)
    #             scaler.update()

    # # 训练完后打印报告
    # print(prof.key_averages().table(
    #     sort_by="cpu_time_total", row_limit=20
    # ))
    # exit(0)
    

    # for data in dataloader:
    #     # start = time.perf_counter()
    #     i += 1
    #     model.zero_grad()
    #     x = data.to(device)
    #     y = data.y.unsqueeze(1).type(torch.float32).to(device)
    #     output, _, _ = model(x)
    #     train_loss = criterion(output,y)
    #     train_loss_sum += train_loss
    #     train_loss.backward()
    #     opt.step()
        
    #     # torch.cuda.synchronize()
    #     # elapsed = time.perf_counter() - start
    #     # print(f"time: {elapsed:.4f} s")
    
    total = 0
    for data in dataloader:
        i += 1
        model.zero_grad()
        x = data.to(device, non_blocking=False)
        y = data.y.unsqueeze(1).type(torch.float32).to(device, non_blocking=False)
        
        # start = time.perf_counter()
        with autocast(device_type = device, dtype=torch.float16):
            output, _, _ = model(x)
            train_loss = criterion(output, y)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start
        # total += elapsed
        
        train_loss_sum += train_loss.detach().item()
        scaler.scale(train_loss).backward()
        scaler.step(opt)
        scaler.update()
        
        # print(f"time: {elapsed:.4f} s")
    # print(f"total time = {total:.6f}")
    train_loss_sum = train_loss_sum
    loss = train_loss_sum/i
    
    return loss,lr_list
        
 

# def train(model,criterion,opt,dataloader,device,args=None,lr_scheduler=None):

#     model.train()
#     train_loss_sum = 0
#     i = 0 
#     lr_list = []

#     for data in dataloader:
#         i += 1
#         model.zero_grad()
#         x = data.to(device)
#         output, _ = model(x)
        
#         if args.task == 'reg':
#             y = data.y.unsqueeze(1).type(torch.float32).to(device)
#             # y = Scalr(y)
#             train_loss = criterion(output,y)
#         if args.task == 'clf':
#             y = data.y_clf.long().to(device)
#             train_loss = criterion(output,y)

#         train_loss_sum += train_loss
#         # opt.optimizer.zero_grad()
#         # opt.zero_grad()
#         train_loss.backward() 

#         if lr_scheduler == 0:           
#             lr = opt.step()
#             # print(f'{i} batch lr: {lr}') 
#             lr_list.append(lr.cpu().detach().item())
#         else:
#             opt.step()
#             lr = opt.param_groups[0]['lr']
#             lr_list.append(lr)  # 返回实时的学习率

#     train_loss_sum = train_loss_sum.cpu().detach().numpy()       
#     loss = train_loss_sum/i
    
#     return loss,lr_list




def validate(model,criterion,dataloader,device,args=None):

    model.eval()
    y_true = []
    y_pred = []
    i = 0
    
    with torch.no_grad():
        for data in dataloader:
            i += 1
            x = data.to(device) 
            y = data.y.unsqueeze(1).to(device)
            y_true.append(y.view(-1, 1))
            output, _, _= model(x)
            y_pred.append(output)            

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()      
    mse,rmse,mae,r2,pearson,spearman = get_metrics(y_true,y_pred)
    m1,m2,m3,m4,m5,m6,m7 = mse,rmse,mae,r2,pearson,spearman,None

    return m1,m2,m3,m4,m5,m6,m7 



def infer(model,dataloader,device,args=None):

    model.eval()
    y_pred = []
    y_true = []
    cell_embed = []
    attn = []
    i = 0
    
    with torch.no_grad():
        for data in dataloader:
            i += 1
            x = data.to(device)
            output, output_cell_embed, output_attn = model(x)
            if args.output_attn:
                cell_embed.append(output_cell_embed)
                attn.append(output_attn)
            y = data.y.unsqueeze(1).to(device)
            y_true.append(y.view(-1, 1))
            y_pred.append(output)
                    

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()      
    mse,rmse,mae,r2,pearson,spearman = get_metrics(y_true,y_pred)
    print('Infer reslut: mse:{:.4f} rmse:{:.4f} mae:{:.4f} r2:{:.4f} pearson:{:.4f} spearman:{:.4f} '.format(mse,rmse,mae,r2,pearson,spearman))
    
    if cell_embed:
        cell_embed = torch.stack(cell_embed, dim=0).cpu().detach().numpy()
    if attn:
        attn = torch.stack(attn, dim=0).cpu().detach().numpy()

    return y_pred, cell_embed, attn 



import argparse
import torch
import random
import numpy as np
from exp.exp_classification_imbalanced import Exp_Classification_Imbalanced

def main():
    parser = argparse.ArgumentParser(description='iTransformer for Imbalanced Classification')
    
    # Basic configuration
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer]')
    
    # Data configuration
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.parquet', help='data file')
    parser.add_argument('--label_col', type=str, default='label', help='label column name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # Task configuration
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options: [classification]')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
    
    # Model configuration
    parser.add_argument('--enc_in', type=int, default=20, help='encoder input size')
    parser.add_argument('--num_class', type=int, default=2, help='number of classes')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    # Loss function configuration
    parser.add_argument('--loss_type', type=str, default='focal',
                        help='loss function: [focal, weighted_ce, cross_entropy]')
    parser.add_argument('--focal_alpha', type=float, default=1.0,
                        help='focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='focal loss gamma parameter')
    parser.add_argument('--pos_weight_scale', type=float, default=1.0,
                        help='additional scaling for positive class weight')
    
    # Optimization configuration
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # Early stopping configuration
    parser.add_argument('--early_stopping_metric', type=str, default='auc_pr',
                        help='metric for early stopping: [auc_pr, auc_roc, f1_score]')
    
    # GPU configuration
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    args = parser.parse_args()
    
    # Set device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    # Set random seed
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    print('Args in experiment:')
    print(args)
    
    # Create experiment
    Exp = Exp_Classification_Imbalanced
    
    for ii in range(args.itr):
        # Setting record of experiments
        setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.seq_len}_' \
                 f'loss{args.loss_type}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_' \
                 f'df{args.d_ff}_eb{args.embed}_dt{args.des}_{ii}'
        
        exp = Exp(args)  # Set experiments
        
        if args.is_training:
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting, test=1)
            
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_classification import Exp_Classification
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.losses import FocalLoss, WeightedCrossEntropyLoss
from utils.metrics_imbalanced import calculate_imbalanced_metrics, plot_imbalanced_metrics

warnings.filterwarnings('ignore')

class Exp_Classification_Imbalanced(Exp_Classification):
    """
    Extended experiment class for imbalanced classification tasks.
    Inherits from the standard classification experiment but adds:
    - Custom loss functions for imbalanced data
    - Comprehensive evaluation metrics
    - Automatic class weight calculation
    """
    
    def __init__(self, args):
        super(Exp_Classification_Imbalanced, self).__init__(args)
        self.class_weights = None
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc_pr': [],
            'val_auc_roc': [],
            'val_f1': []
        }
    
    def _select_criterion(self):
        """
        Select loss function based on args.loss_type.
        Automatically calculates class weights if needed.
        """
        # Get training data to calculate class weights
        train_data, _ = self._get_data(flag='TRAIN')
        
        # Calculate class distribution
        if hasattr(train_data, 'labels'):
            labels = train_data.labels
        else:
            # Collect all labels from the dataset
            labels = []
            for i in range(len(train_data)):
                _, label, _ = train_data[i]
                labels.append(label)
            labels = np.array(labels)
        
        # Calculate class counts and weights
        unique_classes = np.unique(labels)
        class_counts = [np.sum(labels == c) for c in unique_classes]
        self.class_weights = class_counts
        
        print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        print(f"Class ratio: {min(class_counts) / sum(class_counts):.4f}")
        
        # Select loss function
        loss_type = getattr(self.args, 'loss_type', 'cross_entropy')
        
        if loss_type == 'focal':
            # Focal loss parameters
            alpha = getattr(self.args, 'focal_alpha', 1.0)
            gamma = getattr(self.args, 'focal_gamma', 2.0)
            print(f"Using Focal Loss with alpha={alpha}, gamma={gamma}")
            return FocalLoss(alpha=alpha, gamma=gamma, num_classes=self.args.num_class)
            
        elif loss_type == 'weighted_ce':
            # Calculate weights - inverse frequency
            total = sum(class_counts)
            weights = torch.tensor([total / (len(class_counts) * count) 
                                  for count in class_counts], dtype=torch.float)
            
            # Apply additional scaling if specified
            weight_scale = getattr(self.args, 'pos_weight_scale', 1.0)
            if len(weights) == 2:  # Binary classification
                weights[1] *= weight_scale
            
            print(f"Using Weighted Cross-Entropy with weights: {weights.numpy()}")
            return nn.CrossEntropyLoss(weight=weights.to(self.device))
            
        else:
            # Standard cross-entropy
            print("Using standard Cross-Entropy Loss")
            return nn.CrossEntropyLoss()
    
    def vali(self, vali_data, vali_loader, criterion):
        """
        Enhanced validation with imbalanced metrics.
        Returns both loss and comprehensive metrics.
        """
        total_loss = []
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(batch_x, padding_mask, None, None)
                
                loss = criterion(outputs, label.long().squeeze())
                total_loss.append(loss.item())
                
                preds.append(outputs.detach().cpu())
                trues.append(label.cpu())
        
        total_loss = np.average(total_loss)
        
        # Concatenate all predictions and labels
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0).numpy()
        
        # Get probabilities
        probs = torch.softmax(preds, dim=1).numpy()
        
        # Calculate imbalanced metrics
        metrics = calculate_imbalanced_metrics(trues.flatten(), probs)
        
        self.model.train()
        
        return total_loss, metrics
    
    def train(self, setting):
        """
        Enhanced training loop with imbalanced metrics tracking.
        """
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        
        # Early stopping based on AUC-PR instead of accuracy
        early_stopping_metric = getattr(self.args, 'early_stopping_metric', 'auc_pr')
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.001)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            
            # Calculate average training loss
            train_loss = np.average(train_loss)
            
            # Validation with imbalanced metrics
            vali_loss, val_metrics = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics = self.vali(test_data, test_loader, criterion)
            
            # Store metrics history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(vali_loss)
            self.metrics_history['val_auc_pr'].append(val_metrics['auc_pr'])
            self.metrics_history['val_auc_roc'].append(val_metrics['auc_roc'])
            self.metrics_history['val_f1'].append(val_metrics['f1_score'])
            
            # Print comprehensive metrics
            print(f"\nEpoch: {epoch + 1}, Steps: {train_steps}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {vali_loss:.4f} | AUC-PR: {val_metrics['auc_pr']:.4f} | "
                  f"AUC-ROC: {val_metrics['auc_roc']:.4f} | F1: {val_metrics['f1_score']:.4f}")
            print(f"Test Loss: {test_loss:.4f} | AUC-PR: {test_metrics['auc_pr']:.4f} | "
                  f"AUC-ROC: {test_metrics['auc_roc']:.4f} | F1: {test_metrics['f1_score']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")
            
            # Early stopping based on selected metric
            early_stopping_value = -val_metrics[early_stopping_metric]  # Negative because we want to maximize
            early_stopping(early_stopping_value, self.model, path)
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            # Adjust learning rate if using scheduler
            if hasattr(self.args, 'lradj') and self.args.lradj != 'constant':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        # Load best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Save training history and plots
        self._save_training_history(path)
        
        return self.model
    
    def test(self, setting, test=0):
        """
        Enhanced test function with comprehensive imbalanced metrics.
        """
        test_data, test_loader = self._get_data(flag='TEST')
        
        if test:
            print('Loading model...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )
        
        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs = self.model(batch_x, padding_mask, None, None)
                
                preds.append(outputs.detach().cpu())
                trues.append(label.cpu())
        
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0).numpy()
        print(f'Test shape: preds={preds.shape}, trues={trues.shape}')
        
        # Get probabilities
        probs = torch.softmax(preds, dim=1).numpy()
        
        # Calculate comprehensive metrics
        test_metrics = calculate_imbalanced_metrics(trues.flatten(), probs)
        
        # Save results
        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Print detailed results
        print("\n" + "="*50)
        print("TEST SET RESULTS - IMBALANCED CLASSIFICATION")
        print("="*50)
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"AUC-PR: {test_metrics['auc_pr']:.4f}")
        print(f"Optimal Threshold: {test_metrics['optimal_threshold']:.4f}")
        print(f"F1 Score: {test_metrics['f1_score']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        print(f"Specificity: {test_metrics['specificity']:.4f}")
        print("\nConfusion Matrix:")
        print(test_metrics['confusion_matrix'])
        print(f"\nTrue Positives: {test_metrics['true_positives']}")
        print(f"False Positives: {test_metrics['false_positives']}")
        print(f"True Negatives: {test_metrics['true_negatives']}")
        print(f"False Negatives: {test_metrics['false_negatives']}")
        
        # Save metrics plot
        plot_path = os.path.join(folder_path, 'test_metrics_plot.png')
        fig = plot_imbalanced_metrics(test_metrics, save_path=plot_path)
        plt.close(fig)
        
        # Save detailed results to file
        file_name = 'result_classification_imbalanced.txt'
        with open(os.path.join(folder_path, file_name), 'w') as f:
            f.write(f"Experiment: {setting}\n")
            f.write("="*50 + "\n")
            f.write("IMBALANCED CLASSIFICATION RESULTS\n")
            f.write("="*50 + "\n")
            for key, value in test_metrics.items():
                if key != 'confusion_matrix':
                    f.write(f"{key}: {value:.4f}\n")
            f.write(f"\nConfusion Matrix:\n{test_metrics['confusion_matrix']}\n")
            f.write(f"\nOptimal Threshold: {test_metrics['optimal_threshold']:.4f}\n")
        
        # Save metrics as numpy file for later analysis
        np.save(os.path.join(folder_path, 'test_metrics.npy'), test_metrics)
        
        return test_metrics
    
    def _save_training_history(self, path):
        """
        Save training history and create plots.
        """
        import matplotlib.pyplot as plt
        
        # Save history as numpy file
        np.save(os.path.join(path, 'training_history.npy'), self.metrics_history)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss plot
        axes[0, 0].plot(self.metrics_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.metrics_history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC-PR plot
        axes[0, 1].plot(self.metrics_history['val_auc_pr'], label='Val AUC-PR', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC-PR')
        axes[0, 1].set_title('Validation AUC-PR (Primary Metric for Imbalanced Data)')
        axes[0, 1].grid(True)
        
        # AUC-ROC plot
        axes[1, 0].plot(self.metrics_history['val_auc_roc'], label='Val AUC-ROC', color='blue')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC-ROC')
        axes[1, 0].set_title('Validation AUC-ROC')
        axes[1, 0].grid(True)
        
        # F1 score plot
        axes[1, 1].plot(self.metrics_history['val_f1'], label='Val F1', color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'training_history.png'))
        plt.close()

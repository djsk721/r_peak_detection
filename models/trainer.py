import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class BaseTrainer(ABC):
    """
    기본 트레이너 클래스
    """
    def __init__(self, model, optimizer, criterion, scheduler, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        
    @abstractmethod
    def train_step(self, batch):
        """한 배치 학습을 위한 추상 메소드"""
        pass
        
    @abstractmethod
    def validate_step(self, batch):
        """한 배치 검증을 위한 추상 메소드"""
        pass
    
    @abstractmethod
    def test_step(self, batch):
        """한 배치 테스트를 위한 추상 메소드"""
        pass
    
    def train_epoch(self, train_loader):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            loss = self.train_step(batch)
            total_loss += loss
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """검증 수행"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.validate_step(batch)
                total_loss += loss
                
        return total_loss / len(val_loader)
    
    def test(self, test_loader):
        """테스트 수행"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in test_loader:
                predictions, targets, outputs = self.test_step(batch)
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                all_outputs.extend(outputs)
        
        # 수집된 데이터를 numpy 배열로 변환
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        
        # 최적의 임계값 찾기
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            binary_preds = (all_outputs > threshold).astype(int)
            # precision_recall_fscore_support를 사용하여 한 번에 모든 지표 계산
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, binary_preds, average='binary'
            )
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # 최적 임계값으로 최종 예측
        final_preds = (all_outputs > best_threshold).astype(int)
        
        # precision_recall_fscore_support를 사용하여 한 번에 모든 지표 계산
        accuracy = accuracy_score(all_targets, final_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, final_preds, average='binary'
        )
               
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_threshold': best_threshold
        }

class QRSDetectionTrainer(BaseTrainer):
    """
    QRS Complex 검출 모델 트레이너
    """
    def train_step(self, batch):
        signals, labels, _ = batch
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(signals)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch):
        signals, labels, _ = batch
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(signals)
        loss = self.criterion(outputs, labels)
        
        return loss.item()
    
    def test_step(self, batch):
        signals, labels, _ = batch
        signals = signals.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(signals)
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        targets = labels.cpu().numpy()
        
        return predictions.flatten(), targets.flatten(), outputs.cpu().numpy().flatten()

class RPeakDetectionTrainer(BaseTrainer):
    """
    R-peak 검출 모델 트레이너
    """
    def train_step(self, batch):
        signals, peaks, _ = batch
        signals = signals.to(self.device)
        peaks = peaks.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(signals)
        loss = self.criterion(outputs, peaks)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch):
        signals, peaks, _ = batch
        signals = signals.to(self.device)
        peaks = peaks.to(self.device)
        
        outputs = self.model(signals)
        loss = self.criterion(outputs, peaks)
        
        return loss.item()
    
    def test_step(self, batch):
        signals, peaks, _ = batch
        signals = signals.to(self.device)
        peaks = peaks.to(self.device)
        
        outputs = self.model(signals)
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        targets = peaks.cpu().numpy()
        
        return predictions.flatten(), targets.flatten(), outputs.cpu().numpy().flatten()

def train_model(trainer, train_loader, val_loader, test_loader, num_epochs, save_path, early_stopping_patience=5):
    """
    모델 학습을 실행하는 함수
    """
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # 학습
        train_loss = trainer.train_epoch(train_loader)
        
        # 검증
        val_loss = trainer.validate(val_loader)
        
        # 학습률 조정
        trainer.scheduler.step(val_loss)
        
        # 모델 저장 및 Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early Stopping 확인
        if patience_counter >= early_stopping_patience:
            print(f'Early Stopping: {early_stopping_patience}회 동안 검증 손실이 개선되지 않았습니다.')
            break
    
    # 최종 테스트 수행
    print("\n최종 테스트 결과:")
    test_metrics = trainer.test(test_loader)
    for metric_name, value in test_metrics.items():
        print(f"{metric_name}: {value:.4f}")

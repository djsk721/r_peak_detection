import torch
import torch.nn as nn

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3):
        super(CBR, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, padding=3, reduction=16):
        super(ResidualBlock, self).__init__()
        self.cbr1 = CBR(channels, channels, kernel_size, 1, padding)
        self.cbr2 = CBR(channels, channels, kernel_size, 1, padding)
        
      
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), 
            nn.Conv1d(channels, channels // reduction, kernel_size=1),  
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),  
            nn.Sigmoid()  
        )
        
    def forward(self, x):
        residual = x
        x = self.cbr1(x)
        x = self.cbr2(x)
        
        # Squeeze and Excitation 적용
        se_weight = self.se(x)
        x = x * se_weight  # 채널별 가중치 적용
        
        x = x + residual  
        return x

class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(EncodingBlock, self).__init__()
        self.cbr = CBR(in_channels, out_channels, stride=stride)
        self.res_blocks = nn.Sequential(
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )
        
    def forward(self, x):
        x = self.cbr(x)
        x = self.res_blocks(x)
        return x

class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(DecodingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.cbr = CBR(in_channels, out_channels)
        
        # Squeeze-and-Excitation 모듈 추가
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  
            nn.Conv1d(out_channels, out_channels // reduction, kernel_size=1),  
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels // reduction, out_channels, kernel_size=1),  
            nn.Sigmoid() 
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        # 크기 불일치 문제 해결을 위해 skip 연결 전에 크기 확인 및 조정
        if x.size(2) != skip.size(2):
            if x.size(2) > skip.size(2):
                x = x[:, :, :skip.size(2)]
            else:
                skip = skip[:, :, :x.size(2)]
        x = torch.cat([x, skip], dim=1)
        x = self.cbr(x)
        
        # Squeeze-and-Excitation 적용
        se_weight = self.se(x)
        x = x * se_weight  # 채널별 가중치 적용
        
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet, self).__init__()
        
        self.enc1 = EncodingBlock(in_channels, 16)
        self.enc2 = EncodingBlock(16, 32)
        self.enc3 = EncodingBlock(32, 48)
        self.enc4 = EncodingBlock(48, 64)
        
        # 디코더 블록
        self.dec1 = DecodingBlock(64 + 48, 48)
        self.dec2 = DecodingBlock(48 + 32, 32)
        self.dec3 = DecodingBlock(32 + 16, 16)
        
        self.conv_final = nn.Conv1d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        dec1 = self.dec1(enc4, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        
        # 마지막 컨볼루션 레이어로 세그멘테이션 마스크 생성
        output = self.conv_final(dec3)
        
        return output
    

# QRS 복합체 회귀 모델 정의
class RPeakModel(nn.Module):
    """
    - 각 QRS 복합체는 32 길이로 리샘플링됨
    - U-Net과 동일한 컨볼루션 레이어 가중치 크기와 채널 수 사용
    - 2개의 인코딩 블록과 2개의 FC 레이어로 구성
    """
    def __init__(self, in_channels=1):
        super(RPeakModel, self).__init__()
        
        self.enc1 = EncodingBlock(in_channels, 16)
        self.enc2 = EncodingBlock(16, 32)
        
        # 글로벌 평균 풀링 적용
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 회귀를 위한 완전 연결 계층
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 1)  # 회귀 출력을 위한 단일 값
        
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
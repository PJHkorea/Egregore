import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

class AdaptiveTopologyConfig:
    LATENT_DIM: int = 128          # 반드시 짝수여야 토러스 분할 사상이 가능
    INIT_SMOOTH_ALPHA: float = 5.0 # 초기 Sigmoid 경사도
    INIT_THRESHOLD_ETA: float = 0.85 # 초기 코사인 유사도 기준선
    PERTURB_BUBBLE: float = 0.1    # 하이퍼네트워크 변형 한계치

class IndependentTopologyGenerator:
    @staticmethod
    def generate_sphere_anchor(dim: int) -> torch.Tensor:
        """L2 Norm = 1.0을 만족하는 고정 구면 기저"""
        v = torch.ones(dim)
        return F.normalize(v, p=2, dim=0)

    @staticmethod
    def generate_torus_anchor(dim: int) -> torch.Tensor:
        """[피드백 반영] 차원을 절반으로 나누어 독립 위상 고리를 구성하는 진정한 토러스 사상"""
        if dim % 2 != 0:
            raise ValueError("Torus 매핑을 위해 차원은 반드시 짝수여야 합니다.")
        
        half_dim = dim // 2
        # 독립적인 위상 축 생성을 위한 라디안 그리드
        t = torch.linspace(0, 2 * math.pi, half_dim)
        
        # 앞부분은 Cosine, 뒷부분은 Sine으로 분리하여 고차원 고리(Ring) 구조 확립
        torus_vector = torch.cat([torch.cos(t), torch.sin(t)], dim=0)
        return F.normalize(torus_vector, p=2, dim=0)
class ParameterizedTopologyGate(nn.Module):
    """[피드백 반영] alpha와 eta를 nn.Parameter로 등록하여 최적화 가능한 적응형 게이트"""
    def __init__(self):
        super().__init__()
        # 역전파 학습이 가능하도록 nn.Parameter 등록
        self.alpha = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_SMOOTH_ALPHA))
        # eta는 -1 ~ 1 사이를 유지하도록 내부적으로 제어할 예정
        self.raw_eta = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_THRESHOLD_ETA))

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 단위 [B, Dim] 입력을 안전하게 처리하는 전 구간 미분 가능 게이팅"""
        # 1. 배치 차원(-1)을 기준으로 코사인 유사도 연산 수행 -> 출력 크기: [Batch_Size]
        cos_sim = F.cosine_similarity(observer_state, latent_weight, dim=-1)
        
        # 2. eta 파라미터의 안정적인 바운더리 보장 (Tanh를 통한 -1.0 ~ 1.0 제어)
        bounded_eta = torch.tanh(self.raw_eta)
        
        # 3. 데이터에 의해 동적으로 변하는 소프트 게이팅 스코어 계산
        # 안정성을 위해 alpha는 양수 절대값 처리
        gate_score = torch.sigmoid(torch.abs(self.alpha) * (cos_sim - bounded_eta))
        
        return gate_score, cos_sim
class BatchResidualHyperNetwork(nn.Module):
    """배치 차원을 지원하고 호메오스타시스 버블 내에서 잔차를 연산하는 하이퍼네트워크"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh()
        )
        self.bubble_limit = AdaptiveTopologyConfig.PERTURB_BUBBLE

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # 입력이 [B, Dim]이든 [B, S, Dim]이든 마지막 차원을 기준으로 변형 발생
        raw_perturbation = self.net(state)
        return self.bubble_limit * F.normalize(raw_perturbation, p=2, dim=-1)

class ProductionEnergyParityLayer(nn.Module):
    """[피드백 반영 완료] 배치 단위 학습 및 적응형 파라미터 최적화가 가능한 최종 매니폴드 레이어"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = ParameterizedTopologyGate()
        self.hypernet = BatchResidualHyperNetwork(dim)
        
        # 고차원 독립 위상 앵커 등록
        self.register_buffer('sphere_anchor', IndependentTopologyGenerator.generate_sphere_anchor(dim))
        self.register_buffer('torus_anchor', IndependentTopologyGenerator.generate_torus_anchor(dim))

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 입력 차원 정보 획득 (예: 배치 사이즈)
        batch_shape = observer_state.shape[:-1]
        
        # 1. 고정 앵커 가중치를 입력 배치 크기에 맞게 동적으로 확장 (Autograd 브로드캐스팅 보장)
        # [Dim] -> [Batch_Size, Dim]
        expanded_sphere = self.sphere_anchor.expand(*batch_shape, self.dim)
        expanded_torus = self.torus_anchor.expand(*batch_shape, self.dim)
        
        # 2. 배치 단위 게이트 점수 확보 -> gate_score 크기: [Batch_Size]
        gate_score, cos_sim = self.gate(observer_state, expanded_sphere)
        
        # 3. 차원 정렬을 위해 gate_score에 차원 추가 -> [Batch_Size, 1]
        gate_mask = gate_score.unsqueeze(-1)
        
        # 4. 매니폴드 토폴로지 블렌딩
        morphed_topology = (1.0 - gate_mask) * expanded_sphere + gate_mask * expanded_torus
        
        # 5. 하이퍼네트워크 델타 주입
        perturbation = self.hypernet(observer_state)
        final_latent_space = morphed_topology + perturbation
        
        # 6. 배치 차원 전체에 대한 강력한 L2 Norm = 1.0 에너지 보존 제어
        conserved_weights = F.normalize(final_latent_space, p=2, dim=-1)
        
        metrics = {
            "cosine_similarity": cos_sim,
            "gate_score": gate_score,
            "l2_norm": torch.norm(conserved_weights, p=2, dim=-1),
            "learned_alpha": self.gate.alpha,
            "learned_eta": torch.tanh(self.gate.raw_eta)
        }
        return conserved_weights, metrics
def main():
    print("========================================================================")
    print("🌌 Egregore Advanced Engine: Batch Operations & Adaptive Topology Test")
    print("========================================================================")
    
    batch_size = 4
    dim = AdaptiveTopologyConfig.LATENT_DIM
    
    # 레이어 생성 및 최적화 대상 선언
    alignment_layer = ProductionEnergyParityLayer(dim=dim)
    optimizer = torch.optim.Adam(alignment_layer.parameters(), lr=0.05)
    
    print(f"초기 설정 하이퍼파라미터 -> Alpha: {alignment_layer.gate.alpha.item():.2f}, "
          f"Eta: {torch.tanh(alignment_layer.gate.raw_eta).item():.4f}")
    print(f"테스트 주입 배치 크기: [{batch_size}, {dim}] (Batch Pipeline 구현 완료)")
    print("-" * 88)
    
    # 가상의 배치 최적화 3단계 시뮬레이션
    for epoch in range(3):
        # 무작위 배치 데이터 생성 및 정규화
        observer_batch = torch.randn(batch_size, dim)
        observer_batch = F.normalize(observer_batch, p=2, dim=-1)
        
        # 1. Forward Pass (배치 처리)
        weights, metrics = alignment_layer(observer_batch)
        
        # 2. 미분 및 파라미터 변화 유도를 위한 손실 함수 설정
        # (출력 가중치가 특정 배치 목표에 도달하도록 그레디언트 유도)
        target_batch = torch.ones(batch_size, dim) * 0.05
        loss = F.mse_loss(weights, target_batch)
        
        # 3. Backward Pass & Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. 결과 및 학습 가시성 출력
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | "
              f"배치 평균 유사도: {metrics['cosine_similarity'].mean().item():.4f} | "
              f"L2 Norm 보존상태: {metrics['l2_norm'].mean().item():.1f} | "
              f"업데이트된 Alpha: {metrics['learned_alpha'].item():.3f} | "
              f"업데이트된 Eta: {metrics['learned_eta'].item():.3f}")

    print("-" * 88)
    print("✅ 피드백 반영 완료: 배치 단위 연산이 에러 없이 구동되며, 게이팅 파라미터가 역전파를 통해 동적으로 자율 튜닝됩니다.")

if __name__ == "__main__":
    main()

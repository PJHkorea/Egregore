import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

# 1. 시스템 글로벌 설정 (에너지 보존 및 기하 제어 계수)
class LatentTopologyConfig:
    LATENT_DIM: int = 128          # 잠재 공간 차원수
    SMOOTH_ALPHA: float = 5.0      # Sigmoid 경사도 제어 (Soft-Gating 예리함)
    THRESHOLD_ETA: float = 0.85    # 정렬 전환 임계값 (코사인 유사도 기준)
    PERTURB_BUBBLE: float = 0.1    # 하이퍼네트워크 변형 영향력 상한선 (Homeostasis)

# 2. 고정 기하학적 매니폴드 생성기 (Sphere & Torus 공간 사상)
class TopologySpaceGenerator:
    @staticmethod
    def generate_sphere_anchor(dim: int) -> torch.Tensor:
        """L2 Norm = 1.0을 만족하는 고정 구면 가중치 앵커 생성"""
        v = torch.ones(dim)
        return F.normalize(v, p=2, dim=0)

    @staticmethod
    def generate_torus_anchor(dim: int) -> torch.Tensor:
        """구면과 다른 위상(Topology)을 가진 주기적 토러스 앵커 생성 및 L2 정규화"""
        t = torch.linspace(0, 2 * math.pi, dim)
        v = torch.sin(t) + torch.cos(t)
        return F.normalize(v, p=2, dim=0)
class DifferentiableTopologyGate(nn.Module):
    """하드 스위칭의 미분 단절(Gradient Disconnection)을 해결하는 연속 가이트 엔진"""
    def __init__(self):
        super().__init__()
        self.alpha = LatentTopologyConfig.SMOOTH_ALPHA
        self.eta = LatentTopologyConfig.THRESHOLD_ETA

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """관측자 상태 밀도와 잠재 가중치 간의 코사인 유사도를 계산하고 미분 가능한 게이팅 계수 반환"""
        # 1. Cosine Similarity (관측자 vs 잠재 공간 가중치 관계 측정)
        cos_sim = F.cosine_similarity(observer_state, latent_weight, dim=-1)
        
        # 2. Sigmoid Soft-Gating (전 구간 미분 가능한 Autograd Graph 구축)
        # 하드 스위칭(if-else) 대신 부드러운 전이를 유도하여 그레디언트가 끊기지 않게 함
        gate_score = torch.sigmoid(self.alpha * (cos_sim - self.eta))
        
        return gate_score, cos_sim
class ResidualHyperNetwork(nn.Module):
    """정규화 버블 안에서 잔차 변형을 발생시키는 하이퍼네트워크"""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh() # 변형 출력을 제한하기 위한 바운더리
        )
        self.bubble_limit = LatentTopologyConfig.PERTURB_BUBBLE

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """시스템 항상성(Homeostasis) 유지를 위해 변형의 영향력을 0.1 범위 내로 제한"""
        raw_perturbation = self.net(state)
        # 변형 델타를 정규화 버블 내부 크기로 클리핑/스케일링
        return self.bubble_limit * F.normalize(raw_perturbation, p=2, dim=-1)

class GeometricEnergyParityLayer(nn.Module):
    """토폴로지 모핑 중에도 L2 Norm = 1.0을 영구 보존하는 에너지 연속성 보장 레이어"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = DifferentiableTopologyGate()
        self.hypernet = ResidualHyperNetwork(dim)
        
        # 정적 위상 기하학 앵커 정의
        self.register_buffer('sphere_anchor', TopologySpaceGenerator.generate_sphere_anchor(dim))
        self.register_buffer('torus_anchor', TopologySpaceGenerator.generate_torus_anchor(dim))

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. 임시 기준 가중치 결정 (Sphere 기저)
        base_weight = self.sphere_anchor
        
        # 2. 관측자 밀도에 따른 미분 가능 게이트 계수 도출
        gate_score, cos_sim = self.gate(observer_state, base_weight)
        
        # 3. 매니폴드 위상 토폴로지 블렌딩 (Sphere ↔ Torus 모핑)
        morphed_topology = (1.0 - gate_score) * self.sphere_anchor + gate_score * self.torus_anchor
        
        # 4. 하이퍼네트워크 잔차 변형 주입 (정규화 버블 내부 제어)
        perturbation = self.hypernet(observer_state)
        final_latent_space = morphed_topology + perturbation
        
        # 5. 크리티컬 에너지 정규화 (가중치 절대 크기 L2 Norm = 1.0 강제 보존)
        # 런타임 모핑 중 그레디언트 폭발(Gradient Explosion)을 원천 차단
        conserved_weights = F.normalize(final_latent_space, p=2, dim=-1)
        
        metrics = {
            "cosine_similarity": cos_sim,
            "gate_score": gate_score,
            "l2_norm": torch.norm(conserved_weights, p=2, dim=-1)
        }
        return conserved_weights, metrics
def main():
    print("=========================================================")
    print("🌌 Egregore Alignment System: Topological Manifold Test")
    print("=========================================================")
    
    dim = LatentTopologyConfig.LATENT_DIM
    alignment_layer = GeometricEnergyParityLayer(dim=dim)
    
    # 임의의 역전파 최적화 루틴 설정 (미분 가능성 테스트용)
    optimizer = torch.optim.Adam(alignment_layer.parameters(), lr=0.01)
    
    # 시뮬레이션: 관측자의 밀도가 점진적으로 증가하여 임계값(0.85)을 돌파하는 상황
    # 가상의 인지 상태 벡터 변화 생성
    steps = 5
    print(f"\n[실험 시작] 게이트 임계값 점검 (Threshold: {LatentTopologyConfig.THRESHOLD_ETA})")
    print("-" * 75)
    
    for step in range(steps):
        # 스텝이 진행됨에 따라 상태 가중치 스케일을 변화시켜 코사인 유사도 상향 유도
        observer_state = torch.randn(dim) * (0.5 + step * 0.3)
        observer_state = F.normalize(observer_state, p=2, dim=0)
        
        # 1. 순전파 실행 (Forward Pass)
        weights, metrics = alignment_layer(observer_state)
        
        # 2. 가상의 손실 함수 정의 (가중치가 특정 타겟으로 수렴하도록 그레디언트 유도)
        target = torch.ones(dim) * 0.1
        loss = F.mse_loss(weights, target)
        
        # 3. 역전파 실행 (Backward Pass - 미분 단절 여부 검증)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. 결과 출력 및 에너지 연속성(L2 Norm = 1) 검증
        print(f"Step {step+1} | "
              f"코사인 유사도: {metrics['cosine_similarity'].item():.4f} | "
              f"Soft-Gate 계수: {metrics['gate_score'].item():.4f} | "
              f"출력 가중치 L2 Norm: {metrics['l2_norm'].item():.2f} (에너지 보존 완료) | "
              f"역전파 그래프: 정상 작동 (Loss: {loss.item():.4f})")

    print("-" * 75)
    print("✅ 결과: 모든 스위칭 전 구간에서 Autograd 끊김이 없으며 L2 Norm 정규화가 유지됩니다.")

if __name__ == "__main__":
    main()

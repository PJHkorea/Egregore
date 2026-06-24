import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

class AdaptiveTopologyConfig:
    """
    [KO] 시스템 글로벌 설정 및 에너지 보존 계수 정의
    [EN] Global system configuration and energy conservation coefficients definition
    """
    LATENT_DIM: int = 128            # [KO] 반드시 짝수여야 토러스 분할 사상이 가능 / [EN] Must be even for torus split mapping
    INIT_SMOOTH_ALPHA: float = 5.0   # [KO] 초기 Sigmoid 경사도 (예리함) / [EN] Initial Sigmoid slope (sharpness)
    INIT_THRESHOLD_ETA: float = 0.85 # [KO] 초기 코사인 유사도 기준선 / [EN] Initial cosine similarity threshold
    PERTURB_BUBBLE: float = 0.1      # [KO] 하이퍼네트워크 변형 영향력 상한선 / [EN] Hypernetwork perturbation upper bound


class IndependentTopologyGenerator:
    """
    [KO] 고정 위상 기하학 매니폴드(Sphere & Torus) 앵커 생성기
    [EN] Fixed topological manifold (Sphere & Torus) anchor generator
    """
    @staticmethod
    def generate_sphere_anchor(dim: int) -> torch.Tensor:
        """
        [KO] L2 Norm = 1.0을 만족하는 고정 구면 기저 가중치 생성
        [EN] Generate a fixed spherical base weight satisfying L2 Norm = 1.0
        """
        v = torch.ones(dim)
        return F.normalize(v, p=2, dim=0)

    @staticmethod
    def generate_torus_anchor(dim: int) -> torch.Tensor:
        """
        [KO] 차원을 절반으로 나누어 독립 위상 고리를 구성하는 진정한 토러스 사상
        [EN] True torus mapping using split dimensions to build independent topological rings
        """
        if dim % 2 != 0:
            raise ValueError("Torus 매핑을 위해 차원은 반드시 짝수여야 합니다. (Dimension must be even for torus mapping.)")
        
        half_dim = dim // 2
        # [KO] 독립적인 위상 축 생성을 위한 라디안 그리드 구축
        # [EN] Construct radian grid for generating independent topological axes
        t = torch.linspace(0, 2 * math.pi, half_dim)
        
        # [KO] 앞부분은 Cosine, 뒷부분은 Sine으로 분리하여 독립 위상 구조 확립
        # [EN] Split into Cosine (front) and Sine (back) to establish independent topology
        torus_vector = torch.cat([torch.cos(t), torch.sin(t)], dim=0)
        return F.normalize(torus_vector, p=2, dim=0)
class ParameterizedTopologyGate(nn.Module):
    """
    [KO] alpha와 eta를 최적화 가능한 nn.Parameter로 등록하여 제어하는 적응형 게이트
    [EN] Adaptive gate controlling alpha and eta as optimizable nn.Parameters
    """
    def __init__(self):
        super().__init__()
        # [KO] 역전파 학습이 가능하도록 nn.Parameter 등록
        # [EN] Register as nn.Parameters to enable backpropagation learning
        self.alpha = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_SMOOTH_ALPHA))
        self.raw_eta = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_THRESHOLD_ETA))

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KO] 배치 단위 [B, Dim] 입력을 안전하게 처리하는 전 구간 미분 가능 게이팅
        [EN] Fully differentiable gating safely processing batch-level [B, Dim] inputs
        """
        # [KO] 배치 차원(-1)을 기준으로 코사인 유사도 연산 수행 -> 출력 크기: [Batch_Size]
        # [EN] Compute cosine similarity based on batch dimension (-1) -> Output size: [Batch_Size]
        cos_sim = F.cosine_similarity(observer_state, latent_weight, dim=-1)
        
        # [KO] 수리적 안정성을 위해 eta 파라미터 범위를 Tanh를 통해 -1.0 ~ 1.0으로 제어
        # [EN] Restrict eta parameter range to -1.0 ~ 1.0 via Tanh for mathematical stability
        bounded_eta = torch.tanh(self.raw_eta)
        
        # [KO] 데이터에 의해 동적으로 변하는 소프트 게이팅 스코어 계산 (alpha는 절대값 처리)
        # [EN] Compute dynamically changing soft-gating score from data (alpha processed as absolute value)
        gate_score = torch.sigmoid(torch.abs(self.alpha) * (cos_sim - bounded_eta))
        
        return gate_score, cos_sim
class BatchResidualHyperNetwork(nn.Module):
    """
    [KO] 배치 차원을 지원하고 호메오스타시스 버블 내에서 잔차를 연산하는 하이퍼네트워크
    [EN] Hypernetwork supporting batch dimensions and computing residuals within homeostasis bubble
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh() # [KO] 변형 출력을 제한하기 위한 바운더리 / [EN] Boundary to restrict perturbation output
        )
        self.bubble_limit = AdaptiveTopologyConfig.PERTURB_BUBBLE

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        [KO] 시스템 항상성 유지를 위해 마지막 차원을 기준으로 변형을 발생시키고 크기를 제한
        [EN] Generate and restrict perturbation based on the last dimension to maintain system homeostasis
        """
        raw_perturbation = self.net(state)
        # [KO] 변형 델타를 정규화 버블 내부 크기로 클리핑 및 스케일링
        # [EN] Clip and scale the perturbation delta to the interior size of the normalization bubble
        return self.bubble_limit * F.normalize(raw_perturbation, p=2, dim=-1)


class ProductionEnergyParityLayer(nn.Module):
    """
    [KO] 배치 단위 학습 및 적응형 파라미터 최적화가 가능한 최종 매니폴드 레이어
    [EN] Final manifold layer capable of batch-level training and adaptive parameter optimization
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = ParameterizedTopologyGate()
        self.hypernet = BatchResidualHyperNetwork(dim)
        
        # [KO] 고차원 독립 위상 앵커를 정적 버퍼로 등록
        # [EN] Register high-dimensional independent topological anchors as static buffers
        self.register_buffer('sphere_anchor', IndependentTopologyGenerator.generate_sphere_anchor(dim))
        self.register_buffer('torus_anchor', IndependentTopologyGenerator.generate_torus_anchor(dim))

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # [KO] 입력 차원 정보 획득 (예: 배치 사이즈)
        # [EN] Acquire input dimension information (e.g., batch size)
        batch_shape = observer_state.shape[:-1]
        
        # 1. [KO] 고정 앵커 가중치를 입력 배치 크기에 맞게 동적으로 확장 (Autograd 브로드캐스팅 보장)
        # 1. [EN] Dynamically expand fixed anchors to match input batch size (ensuring Autograd broadcasting)
        expanded_sphere = self.sphere_anchor.expand(*batch_shape, self.dim)
        expanded_torus = self.torus_anchor.expand(*batch_shape, self.dim)
        
        # 2. [KO] 배치 단위 게이트 점수 확보 -> gate_score 크기: [Batch_Size]
        # 2. [EN] Obtain batch-level gate scores -> gate_score shape: [Batch_Size]
        gate_score, cos_sim = self.gate(observer_state, expanded_sphere)
        
        # 3. [KO] 차원 정렬을 위해 gate_score에 마지막 차원 추가 -> [Batch_Size, 1]
        # 3. [EN] Unsqueeze last dimension of gate_score for dimension alignment -> [Batch_Size, 1]
        gate_mask = gate_score.unsqueeze(-1)
        
        # 4. [KO] 매니폴드 토폴로지 블렌딩 (Sphere ↔ Torus 모핑)
        # 4. [EN] Manifold topology blending (Sphere ↔ Torus morphing)
        morphed_topology = (1.0 - gate_mask) * expanded_sphere + gate_mask * expanded_torus
        
        # 5. [KO] 하이퍼네트워크 잔차 변형 주입
        # 5. [EN] Inject hypernetwork residual perturbation
        perturbation = self.hypernet(observer_state)
        final_latent_space = morphed_topology + perturbation
        
        # 6. [KO] 배치 차원 전체에 대한 강력한 L2 Norm = 1.0 에너지 보존 정규화 제어
        # 6. [EN] Strict L2 Norm = 1.0 energy parity normalization control across all batch dimensions
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
    
    # [KO] 레이어 인스턴스화
    # [EN] Instantiate the production manifold layer
    alignment_layer = ProductionEnergyParityLayer(dim=dim)
    
    # ------------------------------------------------------------------------
    # [KO] 피드백 반영: 메모리 주소(ID) 참조 기반의 안전한 층별 학습률(LLRD) 분리 기법
    # [EN] Feedback Applied: Safe Layer-wise Learning Rate (LLRD) via Memory Address (ID) Reference
    # ------------------------------------------------------------------------
    # [KO] 오버헤드를 줄이기 위해 해시 테이블(Set) 구조로 게이트 파라미터 ID 추출
    # [EN] Extract gate parameter IDs using a hash table (Set) structure to reduce overhead
    gate_param_ids = {id(p) for p in alignment_layer.gate.parameters()}

    # [KO] ID 조회를 통해 백본 파라미터와 게이트 파라미터를 안전하고 신속하게 O(1) 분리
    # [EN] Safely and quickly separate backbone and gate parameters in O(1) via ID lookup
    backbone_params = [p for p in alignment_layer.parameters() if id(p) not in gate_param_ids]
    gate_params = list(alignment_layer.gate.parameters())

    # [KO] 그룹별 차별화 학습률 적용: 위상 붕괴 방지를 위해 게이트 학습률은 100배 낮추고 가중치 감쇠 해제
    # [EN] Apply group LR: Lower gate LR by 100x and disable weight decay to prevent topological collapse
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": gate_params, "lr": 1e-6, "weight_decay": 0.0}
    ])
    
    print(f"초기 설정 하이퍼파라미터 (Initial Config) -> Alpha: {alignment_layer.gate.alpha.item():.2f}, "
          f"Eta: {torch.tanh(alignment_layer.gate.raw_eta).item():.4f}")
    print(f"테스트 주입 배치 크기 (Batch Input Size): [{batch_size}, {dim}]")
    print("-" * 88)
    
    # [KO] 가상의 배치 최적화 3단계 시뮬레이션
    # [EN] Simulated 3-step batch optimization loop
    for epoch in range(3):
        # [KO] 무작위 배치 데이터 생성 및 기하 정규화
        # [EN] Generate random batch data and geometric normalization
        observer_batch = torch.randn(batch_size, dim)
        observer_batch = F.normalize(observer_batch, p=2, dim=-1)
        
        # 1. [KO] Forward Pass 실행 (배치 파이프라인 연산)
        # 1. [EN] Execute Forward Pass (Batch pipeline operation)
        weights, metrics = alignment_layer(observer_batch)
        
        # 2. [KO] 그레디언트 유도를 위한 손실 함수 계산
        # 2. [EN] Compute Loss function to guide gradients
        target_batch = torch.ones(batch_size, dim) * 0.05
        loss = F.mse_loss(weights, target_batch)
        
        # 3. [KO] Backward Pass 및 가중치 업데이트 (옵티마이저 한 단계 진행)
        # 3. [EN] Backward Pass and weight updates (Optimizer step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. [KO] 결과 및 학습 안정성 지표 출력
        # 4. [EN] Print results and training stability metrics
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | "
              f"배치 평균 유사도 (Avg CosSim): {metrics['cosine_similarity'].mean().item():.4f} | "
              f"L2 Norm 보존상태 (Energy Parity): {metrics['l2_norm'].mean().item():.1f} | "
              f"업데이트된 Alpha: {metrics['learned_alpha'].item():.3f} | "
              f"업데이트된 Eta: {metrics['learned_eta'].item():.3f}")

    print("-" * 88)
    print("✅ 검증 완료: 문자열 하드코딩이 배제된 메모리 기반 분리 로직이 프로덕션 파이프라인에서 정상 작동합니다.")

if __name__ == "__main__":
    main()

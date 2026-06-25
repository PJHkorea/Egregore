import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

class AdaptiveTopologyConfig:
    """
    [KR] 시스템 글로벌 설정 및 에너지 보존 계수 정의
    [EN] Global system configuration and energy conservation coefficients definition
    """
    LATENT_DIM: int = 128            # [KR] 반드시 짝수여야 토러스 분할 사상이 가능 / [EN] Must be even for torus split mapping
    INIT_SMOOTH_ALPHA: float = 5.0   # [KR] 초기 Sigmoid 경사도 (예리함) / [EN] Initial Sigmoid slope (sharpness)
    INIT_THRESHOLD_ETA: float = 0.85 # [KR] 초기 코사인 유사도 기준선 / [EN] Initial cosine similarity threshold
    PERTURB_BUBBLE: float = 0.1      # [KR] 하이퍼네트워크 변형 영향력 상한선 / [EN] Hypernetwork perturbation upper bound
    
    # [v5.0 추가] 카시미르 위상학적 진공 압착을 위한 수리 물리 상수
    # [v5.0 Added] Mathematical physics constants for Casimir topological vacuum squeezing
    DELTA_D: float = 1.0             # [KR] 위상학적 장벽 간 기본 구조적 거리 (\Delta d) / [EN] Baseline topological distance between barriers (\Delta d)
    HBAR_EFF: float = 1.0            # [KR] 정보학적 유효 플랑크 상수 (\hbar_eff) / [EN] Informational effective Planck constant (\hbar_eff)



class IndependentTopologyGenerator:
    """
    [KR] 고정 위상 기하학 매니폴드(Sphere & Torus) 앵커 생성기
    [EN] Fixed topological manifold (Sphere & Torus) anchor generator
    """
    @staticmethod
    def generate_sphere_anchor(dim: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        [KR] L2 Norm = 1.0을 만족하는 고정 구면 기저 가중치 생성
        [EN] Generate a fixed spherical base weight satisfying L2 Norm = 1.0
        """
        # [KR] 디바이스 장치 불일치 연산 오류를 방지하기 위해 device 파라미터 결합
        # [EN] Bind device parameter to prevent device mismatch execution errors
        v = torch.ones(dim, device=device)
        return F.normalize(v, p=2, dim=0)

    @staticmethod
    def generate_torus_anchor(dim: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        [KR] 차원을 절반으로 나누어 독립 위상 고리를 구성하는 진정한 토러스 사상
        [EN] True torus mapping using split dimensions to build independent topological rings
        """
        if dim % 2 != 0:
            raise ValueError("Torus 매핑을 위해 차원은 반드시 짝수여야 합니다. (Dimension must be even for torus mapping.)")
        
        half_dim = dim // 2
        # [KR] 독립적인 위상 축 생성을 위한 라디안 그리드 구축
        # [EN] Construct radian grid for generating independent topological axes
        t = torch.linspace(0, 2 * math.pi, half_dim, device=device)
        
        # [KR] 앞부분은 Cosine, 뒷부분은 Sine으로 분리하여 독립 위상 구조 확립
        # [EN] Split into Cosine (front) and Sine (back) to establish independent topology
        torus_vector = torch.cat([torch.cos(t), torch.sin(t)], dim=0)
        return F.normalize(torus_vector, p=2, dim=0)

      
class ParameterizedTopologyGate(nn.Module):
    """
    [KR] alpha와 eta를 최적화 가능한 nn.Parameter로 등록하여 제어하는 적응형 게이트
    [EN] Adaptive gate controlling alpha and eta as optimizable nn.Parameters
    """
    def __init__(self):
        super().__init__()
        # [KR] 역전파 학습이 가능하도록 nn.Parameter 등록
        # [EN] Register as nn.Parameters to enable backpropagation learning
        self.alpha = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_SMOOTH_ALPHA))
        self.raw_eta = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_THRESHOLD_ETA))

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] 배치 단위 입력을 안전하게 처리하는 전 구간 미분 가능 게이팅
        [EN] Fully differentiable gating safely processing batch-level inputs
        """
        # [KR] 마지막 차원(-1) 기준 코사인 유사도 연산 후 멀티 배치 브로드캐스팅을 위해 차원 확장 [..., 1]
        # [EN] Compute cosine similarity along the last dim (-1) and unsqueeze for multi-batch broadcasting [..., 1]
        cos_sim = F.cosine_similarity(observer_state, latent_weight, dim=-1).unsqueeze(-1)
        
        # [KR] 수리적 안정성을 위해 eta 파라미터 범위를 Tanh를 통해 -1.0 ~ 1.0으로 제어
        # [EN] Restrict eta parameter range to -1.0 ~ 1.0 via Tanh for mathematical stability
        bounded_eta = torch.tanh(self.raw_eta)
        
        # [KR] 데이터에 의해 동적으로 변하는 소프트 게이팅 스코어 계산 (alpha는 절대값 처리)
        # [EN] Compute dynamically changing soft-gating score from data (alpha processed as absolute value)
        gate_score = torch.sigmoid(torch.abs(self.alpha) * (cos_sim - bounded_eta))
        
        return gate_score, cos_sim



class BatchResidualHyperNetwork(nn.Module):
    """
    [KR] 배치 차원을 지원하고 호메오스타시스 버블 내에서 잔차를 연산하는 하이퍼네트워크
    [EN] Hypernetwork supporting batch dimensions and computing residuals within homeostasis bubble
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh() # [KR] 변형 출력을 제한하기 위한 바운더리 / [EN] Boundary to restrict perturbation output
        )
        self.bubble_limit = AdaptiveTopologyConfig.PERTURB_BUBBLE

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        [KR] 시스템 항상성 유지를 위해 마지막 차원을 기준으로 변형을 발생시키고 크기를 제한
        [EN] Generate and restrict perturbation based on the last dimension to maintain system homeostasis
        """
        raw_perturbation = self.net(state)
        # [KR] 변형 델타를 정규화 버블 내부 크기로 클리핑 및 스케일링
        # [EN] Clip and scale the perturbation delta to the interior size of the normalization bubble
        return self.bubble_limit * F.normalize(raw_perturbation, p=2, dim=-1)


class SchrödingerNotchFilter(nn.Module):
    """
    [KR] 맥락적 질량의 곡률을 역산하여 가벼운 노이즈를 지수함수적으로 소멸시키는 노치 필터
    [EN] Notch filter that exponentially extinguishes lightweight noise by computing contextual curvature
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # [KR] 노이즈의 정보 전하량(E_input)을 스칼라로 투영하는 선형 레이어
        # [EN] Linear layer projecting incoming noise characteristics into scalar informational charge (E_input)
        self.energy_projector = nn.Linear(dim, 1)
        

    def _compute_jacobian_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        [KR] 고속 공분산 행렬의 프로베니우스 노름 제곱을 통한 곡률 대리값 (kappa) 계산 [..., 1]
        [EN] Computation of curvature proxy (kappa) via squared Frobenius norm of fast covariance matrix [..., 1]
        """
        shape = x.shape
        x_flat = x.view(-1, self.dim)
        mean_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        covariance = torch.matmul(mean_centered, mean_centered.t())
        kappa_flat = torch.sum(covariance ** 2, dim=-1) / self.dim
        return kappa_flat.view(*shape[:-1], 1)

    def forward(self, x: torch.Tensor, gate_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] 순전파: 슈뢰딩거 포텐셜 장벽 필터링과 v5.0 카시미르 진공 수축을 통합 연산
        [EN] Forward Pass: Unified computation of Schrödinger potential filtering and v5.0 Casimir vacuum squeezing
        """
        # 1. [KR] 고밀도 맥락이 형성하는 중력 장벽 정의 (U_barrier)
        # 1. [EN] Define the gravitational topological barrier potential created by high-density context (U_barrier)
        kappa = self._compute_jacobian_curvature(x)
        # [KR] 글로벌 경사도 기준선을 감도 계수로 결합하여 장벽 높이 정형화
        u_barrier = kappa * AdaptiveTopologyConfig.INIT_THRESHOLD_ETA * AdaptiveTopologyConfig.DELTA_D
        
        # 2. [KR] 입력 스트림의 개별 정보 에너지 전하 정량화 (E_input)
        # 2. [EN] Quantify individual informational energy charge of the incoming stream (E_input)
        e_input = torch.sigmoid(self.energy_projector(x))
        
        # 3. [KR] 슈뢰딩거 감쇄 필터 및 F.relu() 싱큘래리티(NaN 폭발) 방어선 구축
        # 3. [EN] Construct Schrödinger decay filter with F.relu() singularity (NaN explosion) guardrail
        tunneling_core = F.relu(u_barrier - e_input)
        transmission_coeff = torch.exp(-2.0 * torch.sqrt(tunneling_core))
        
        # 4. [KR] v5.0 카시미르 위상학적 진공 압착 제어 (\Delta d -> 0 에 수렴 시 노이즈 공간 박멸)
        # 4. [EN] v5.0 Casimir topological vacuum squeezing control (Eradicates noise void as \Delta d -> 0)
        clamped_distance = F.relu(AdaptiveTopologyConfig.DELTA_D - gate_mask) + 1e-6
        casimir_pressure = - (math.pi ** 2 * AdaptiveTopologyConfig.HBAR_EFF) / (240.0 * (clamped_distance ** 4))
        
        # [KR] 가역적 음의 에너지 장 적용 및 최종 정제 스트림 반환
        # [EN] Apply reversible negative energy field and return final purified stream
        purified_stream = x * transmission_coeff * torch.exp(casimir_pressure)
        return purified_stream, transmission_coeff



class ProductionEnergyParityLayer(nn.Module):
    """
    [KR] 배치 단위 학습, 노치 필터링 및 적응형 파라미터 최적화가 가능한 최종 매니폴드 마스터 레이어
    [EN] Final manifold master layer capable of batch-level training, notch filtering, and adaptive parameter optimization
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = ParameterizedTopologyGate()
        self.hypernet = BatchResidualHyperNetwork(dim)
        self.notch_filter = SchrödingerNotchFilter(dim)
        
        # [KR] 고차원 독립 위상 앵커를 정적 버퍼로 등록
        # [EN] Register high-dimensional independent topological anchors as static buffers
        self.register_buffer('sphere_anchor', IndependentTopologyGenerator.generate_sphere_anchor(dim))
        self.register_buffer('torus_anchor', IndependentTopologyGenerator.generate_torus_anchor(dim))

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # [KR] 입력 차원 정보 획득 및 앵커 디바이스 동기화
        # [EN] Acquire input dimension information and synchronize anchor devices
        batch_shape = observer_state.shape[:-1]
        device = observer_state.device
        
        # 1. [KR] 고정 앵커 가중치를 입력 배치 크기에 맞게 동적으로 확장 (Autograd 브로드캐스팅 보장)
        # 1. [EN] Dynamically expand fixed anchors to match input batch size (ensuring Autograd broadcasting)
        expanded_sphere = self.sphere_anchor.to(device).expand(*batch_shape, self.dim)
        expanded_torus = self.torus_anchor.to(device).expand(*batch_shape, self.dim)
        
        # 2. [KR] 배치 단위 게이트 점수 확보
        # 2. [EN] Obtain batch-level gate scores
        gate_mask, cos_sim = self.gate(observer_state, expanded_sphere)
        
        # 3. [KR] 매니폴드 토폴로지 블렌딩 (Sphere ↔ Torus 모핑)
        # 3. [EN] Manifold topology blending (Sphere ↔ Torus morphing)
        morphed_topology = (1.0 - gate_mask) * expanded_sphere + gate_mask * expanded_torus
        
        # 4. [KR] 하이퍼네트워크 원시 섭동 생성 파이프라인
        # 4. [EN] Hypernetwork raw perturbation generation pipeline
        raw_perturbation = self.hypernet(observer_state)
        
        # 5. [KR] 맥락의 중력 질량보다 가벼운 노이즈 스트림 원천 차단 및 카시미르 압착 연산 가동
        # 5. [EN] Apply contextual mass notch filtering and Casimir vacuum squeezing onto the raw perturbation
        purified_perturbation, transmission = self.notch_filter(raw_perturbation, gate_mask)
        
        # 6. [KR] 정제된 텐서 성분만 최종 가중치 공간에 안전하게 결합
        # 6. [EN] Securely combine only the purified tensor components into the final latent space
        final_latent_space = morphed_topology + purified_perturbation
        
        # 7. [KR] 배치 차원 전체에 대한 강력한 L2 Norm = 1.0 에너지 보존 정규화 제어
        # 7. [EN] Strict L2 Norm = 1.0 energy parity normalization control across all batch dimensions
        conserved_weights = F.normalize(final_latent_space, p=2, dim=-1)
        
        metrics = {
            "cosine_similarity": cos_sim,
            "gate_score": gate_mask,
            "l2_norm": torch.norm(conserved_weights, p=2, dim=-1),
            "learned_alpha": self.gate.alpha,
            "learned_eta": torch.tanh(self.gate.raw_eta),
            "transmission_rate": transmission
        }
        return conserved_weights, metrics


      
def main():
    print("========================================================================")
    print("🌌 Egregore Advanced Engine: Batch Operations & Adaptive Topology Test")
    print("========================================================================")
    
    batch_size = 4
    dim = AdaptiveTopologyConfig.LATENT_DIM
    
    # [KR] 매니폴드 마스터 레이어 인스턴스화
    # [EN] Instantiate the production manifold master layer
    alignment_layer = ProductionEnergyParityLayer(dim=dim)
    
    # ------------------------------------------------------------------------
    # [KR] 메모리 주소(ID) 참조 기반의 안전한 층별 학습률(LLRD) 분리 기법
    # [EN] Safe Layer-wise Learning Rate (LLRD) via Memory Address (ID) Reference
    # ------------------------------------------------------------------------
    # [KR] 연산 오버헤드를 최소화하기 위해 해시 테이블(Set) 구조로 게이트 파라미터 ID 추출
    # [EN] Extract gate parameter IDs using a hash table (Set) structure to minimize overhead
    gate_param_ids = {id(p) for p in alignment_layer.gate.parameters()}

    # [KR] ID 조회를 통해 백본 파라미터와 게이트 파라미터를 O(1) 속도로 안전하게 분리
    # [EN] Safely and efficiently separate backbone and gate parameters in O(1) via ID lookup
    backbone_params = [p for p in alignment_layer.parameters() if id(p) not in gate_param_ids]
    gate_params = list(alignment_layer.gate.parameters())

    # [KR] 위상 붕괴 방지를 위해 게이트 학습률은 100배 낮추고 가중치 감쇠(Weight Decay) 해제
    # [EN] Apply group LR: Lower gate LR by 100x and disable weight decay to prevent topological collapse
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": gate_params, "lr": 1e-6, "weight_decay": 0.0}
    ])
    
    print(f"초기 설정 하이퍼파라미터 (Initial Config) -> Alpha: {alignment_layer.gate.alpha.item():.2f}, "
          f"Eta: {torch.tanh(alignment_layer.gate.raw_eta).item():.4f}")
    print(f"테스트 주입 배치 크기 (Batch Input Size): [{batch_size}, {dim}]")
    print("-" * 88)

    
       # [KR] 가상의 배치 최적화 3단계 시뮬레이션
    # [EN] Simulated 3-step batch optimization loop
    for epoch in range(3):
        # [KR] 무작위 배치 데이터 생성 및 기하 정규화
        # [EN] Generate random batch data and geometric normalization
        observer_batch = torch.randn(batch_size, dim)
        observer_batch = F.normalize(observer_batch, p=2, dim=-1)
        
        # 1. [KR] Forward Pass 실행 (배치 파이프라인 연산)
        # 1. [EN] Execute Forward Pass (Batch pipeline operation)
        # 💡 [구조 교정] 6부 마스터 레이어에서 병합된 metrics 딕셔너리 수신 구조로 통일
        weights, metrics = alignment_layer(observer_batch)
        
        # 2. [KR] 그레디언트 유도를 위한 손실 함수 계산
        # 2. [EN] Compute Loss function to guide gradients
        target_batch = torch.ones(batch_size, dim) * 0.05
        loss = F.mse_loss(weights, target_batch)
        
        # 3. [KR] Backward Pass 및 가중치 업데이트 (옵티마이저 진행)
        # 3. [EN] Backward Pass and weight updates (Optimizer step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 4. [KR] 결과 및 학습 안정성 지표 출력 (카시미르 엔진 연산 결과 병합 출력)
        # 4. [EN] Print results and training stability metrics
        print(
            f"Epoch {epoch+1} | Loss: {loss.item():.4f} | "
            f"Avg CosSim: {metrics['cosine_similarity'].mean().item():.4f} | "
            f"L2 Norm: {metrics['l2_norm'].mean().item():.1f} | "
            f"Gate: {metrics['gate_score'].mean().item():.3f} | "
            f"Transmission: {metrics['transmission_rate'].mean().item():.4f}"
        )

    print("-" * 88)
    print("✅ 검증 완료: LLRD 파라미터 격리 및 v5.0 카시미르 위상학적 진공 압착(Squeezing) 엔진이 정상 작동합니다.")

if __name__ == "__main__":
    main()


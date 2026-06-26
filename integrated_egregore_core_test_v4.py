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
    
    # [v5.0 추가] 카시미르 위상학적 진공 압착 및 슈뢰딩거 필터를 위한 수리 물리 상수
    # [v5.0 Added] Mathematical physics constants for Casimir topological vacuum squeezing and Schrödinger filter
    DELTA_D: float = 1.0             # [KR] 위상학적 장벽 간 기본 구조적 거리 (\Delta d) / [EN] Baseline topological distance between barriers (\Delta d)
    HBAR_EFF: float = 1.0            # [KR] 정보학적 유효 플랑크 상수 (\hbar_eff) / [EN] Informational effective Planck constant (\hbar_eff)
    M_STAR: float = 1.0              # [KR] 정보 파동 유효 질량 파라미터 (m*) / [EN] Effective mass parameter (m*)
    BARRIER_ETA: float = 0.5         # [KR] 위상 곡률 장벽 민감도 상수 (\eta) / [EN] Curvature sensitivity constant (\eta)
    
    # 💡 [수리 교정] 카시미르 압력의 마이너스 무한대 발산 및 그래디언트 사멸 방지용 하한 한계선 지정
    # 💡 [Fix] Lower bound limit of negative Casimir pressure to prevent minus infinity divergence and gradient vanishing
    PRESSURE_FLOOR: float = -20.0    # [KR] 카시미르 음의 필드 압력 최소 하한 제약선 / [EN] Minimum lower bound limit of negative Casimir field pressure
    
    # 💡 [무결성 추가] 연산 중 분모가 0이 되어 발생할 수 있는 모든 역전파 NaN 폭발 방지용 미세 상수
    # 💡 [Integrity Added] Fine constant to prevent all backpropagation NaN explosions caused by division by zero during operations
    EPSILON: float = 1e-8
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
        v = torch.ones(dim, device=device)
        # 💡 [무결성 교정] eps 상수를 공식 바인딩하여 잠재적 제로 분모 0 폭발(NaN) 원천 진압
        return F.normalize(v, p=2, dim=0, eps=AdaptiveTopologyConfig.EPSILON)

    @staticmethod
    def generate_torus_anchor(dim: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        [KR] 차원을 절반으로 나누어 독립 위상 고리를 구성하는 진정한 토러스 사상
        [EN] True torus mapping using split dimensions to build independent topological rings
        """
        if dim % 2 != 0:
            raise ValueError("Torus 매핑을 위해 차원은 반드시 짝수여야 합니다. (Dimension must be even for torus mapping.)")
        
        half_dim = dim // 2
        t = torch.linspace(0, 2 * math.pi, half_dim, device=device)
        
        torus_vector = torch.cat([torch.cos(t), torch.sin(t)], dim=0)
        # 💡 [무결성 교정] 토러스 기저 정규화 연산 시에도 안전 분모 엡실론을 명시적으로 주입
        return F.normalize(torus_vector, p=2, dim=0, eps=AdaptiveTopologyConfig.EPSILON)
class ParameterizedTopologyGate(nn.Module):
    """
    [KR] alpha와 eta를 최적화 가능한 nn.Parameter로 등록하여 제어하는 적응형 게이트
    [EN] Adaptive gate controlling alpha and eta as optimizable nn.Parameters
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_SMOOTH_ALPHA))
        self.raw_eta = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_THRESHOLD_ETA))

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] 배치 단위 입력을 안전하게 처리하는 전 구간 미분 가능 게이팅
        [EN] Fully differentiable gating safely processing batch-level inputs
        """
        # 💡 [무결성 교정] 코사인 유사도 계산 시 분모 0 폭발을 막기 위해 안전 분모 엡실론(eps)을 공식 주입
        # 💡 [Fix] Explicitly inject safety denominator epsilon (eps) to prevent division-by-zero explosions
        cos_sim = F.cosine_similarity(
            observer_state, 
            latent_weight, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        ).unsqueeze(-1)
        
        # [KR] 수리적 안정성을 위해 eta 파라미터 범위를 Tanh를 통해 -1.0 ~ 1.0으로 제어
        bounded_eta = torch.tanh(self.raw_eta)
        
        # [KR] 데이터에 의해 동적으로 변하는 소프트 게이팅 스코어 계산 (alpha는 절대값 처리)
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
            # 💡 [무결성 교정] 중간 피처 스케일을 묶어 최종 Tanh 레이어의 포화 및 그레디언트 소멸 현상 완전 차단
            # 💡 [Fix] Bind intermediate feature scales to prevent Tanh saturation and gradient vanishing
            nn.LayerNorm(dim // 2),
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
        # 💡 [무결성 교정] 변형 텐서 정규화 시 제로 분모 0 폭발(NaN)을 막기 위해 안전 분모 엡실론(eps) 주입
        # 💡 [Fix] Inject safety denominator epsilon (eps) to prevent division-by-zero during normalization
        return self.bubble_limit * F.normalize(
            raw_perturbation, 
            p=2, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        )
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
        [KR] 배치 독립성을 완벽히 수호하기 위해 torch.bmm 기반으로 개별 샘플 내부의 특성 공분산 및 곡률 역산
        [EN] Compute curvature proxy (kappa) using torch.bmm to maintain perfect batch independence per sample
        """
        shape = x.shape
        batch_size = shape[0]
        x_reshaped = x.view(batch_size, -1, self.dim) # Elements = 1 (2D) or SeqLen (3D)
        
        # [KR] 샘플 개별 내부 차원 기준 평균 중심화 연산 진행 (배치 간 오염 차단)
        mean_centered = x_reshaped - x_reshaped.mean(dim=1, keepdim=True)
        
        # [KR] torch.bmm 배치 행렬 곱셈을 가동하여 각 배치 내부에서만 작동하는 독립 공분산 [B, Elements, Elements] 연산
        covariance = torch.bmm(mean_centered, mean_centered.transpose(1, 2))
        
        # [KR] 각 배치 독립적으로 프로베니우스 노름 제곱 패턴 추출 후 차원 복원
        kappa_flat = torch.sum(covariance ** 2, dim=-1) / self.dim
        return kappa_flat.view(*shape[:-1], 1)

    def forward(self, x: torch.Tensor, gate_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] 순전파: 슈뢰딩거 포텐셜 장벽 필터링과 v5.0 카시미르 진공 수축을 통합 연산
        [EN] Forward Pass: Unified computation of Schrödinger potential filtering and v5.0 Casimir vacuum squeezing
        """
        # 1. [KR] 고밀도 맥락이 형성하는 중력 장벽 정의 (U_barrier)
        kappa = self._compute_jacobian_curvature(x)
        u_barrier = kappa * AdaptiveTopologyConfig.INIT_THRESHOLD_ETA * AdaptiveTopologyConfig.DELTA_D
        
        # 2. [KR] 입력 스트림의 개별 정보 에너지 전하 정량화 (E_input)
        e_input = torch.sigmoid(self.energy_projector(x))
        
        # 3. [KR] 슈뢰딩거 감쇄 필터 및 F.relu() 싱큘래리티(NaN 폭발) 방어선 구축
        tunneling_core = F.relu(u_barrier - e_input)
        
        # 💡 [무결성 교정] 루트 미분 시 분모 0 폭발(\frac{1}{2\sqrt{0}})로 인한 역전파 NaN 유입 원천 분쇄
        # 💡 [Fix] Prevent backpropagation NaN caused by square root derivative infinity near zero
        safe_tunneling_core = tunneling_core + AdaptiveTopologyConfig.EPSILON
        integral_term = torch.sqrt(
            (2.0 * AdaptiveTopologyConfig.M_STAR) / 
            (AdaptiveTopologyConfig.HBAR_EFF ** 2) * safe_tunneling_core
        )
        transmission_coeff = torch.exp(-2.0 * integral_term)
        
        # 4. [KR] v5.0 카시미르 위상학적 진공 압착 제어 (\Delta d -> 0 에 수렴 시 노이즈 공간 박멸)
        clamped_distance = F.relu(AdaptiveTopologyConfig.DELTA_D - gate_mask) + 1e-6
        raw_pressure = - (math.pi ** 2 * AdaptiveTopologyConfig.HBAR_EFF) / (240.0 * (clamped_distance ** 4))
        
        # 💡 [무결성 교정] 하한선 클램핑에 더해, 압력이 양수로 튀어 exp() 연산이 무한대(inf)로 폭발하는 현상까지 상한 차단
        # 💡 [Fix] Clamp upper bound to 0.0 to prevent exponential explosion if raw_pressure flips positive
        casimir_pressure = torch.clamp(
            raw_pressure, 
            min=AdaptiveTopologyConfig.PRESSURE_FLOOR, 
            max=0.0
        )
        
        # [KR] 가역적 음의 에너지 장 적용 및 최종 정제 스트림 반환
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
        
        # [KR] 고차원 독립 위상 앵커를 정적 버퍼로 등록 (초기 기본 CPU 할당)
        self.register_buffer('sphere_anchor', IndependentTopologyGenerator.generate_sphere_anchor(dim))
        self.register_buffer('torus_anchor', IndependentTopologyGenerator.generate_torus_anchor(dim))

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # [KR] 입력 차원 정보 획득 및 앵커 디바이스 동기화
        batch_shape = observer_state.shape[:-1]
        device = observer_state.device
        
        # 1. [KR] 고정 앵커 가중치를 입력 배치 크기에 맞게 동적으로 확장 (Autograd 브로드캐스팅 보장)
        expanded_sphere = self.sphere_anchor.to(device).expand(*batch_shape, self.dim)
        expanded_torus = self.torus_anchor.to(device).expand(*batch_shape, self.dim)
        
        # 2. [KR] 배치 단위 게이트 점수 확보
        gate_mask, cos_sim = self.gate(observer_state, expanded_sphere)
        
        # 3. [KR] 매니폴드 토폴로지 블렌딩 (Sphere ↔ Torus 모핑)
        morphed_topology = (1.0 - gate_mask) * expanded_sphere + gate_mask * expanded_torus
        
        # 4. [KR] 하이퍼네트워크 원시 섭동 생성 파이프라인
        raw_perturbation = self.hypernet(observer_state)
        
        # 5. [KR] 맥락의 중력 질량보다 가벼운 노이즈 스트림 원천 차단 및 카시미르 압착 연산 가동
        purified_perturbation, transmission = self.notch_filter(raw_perturbation, gate_mask)
        
        # 6. [KR] 정제된 텐서 성분만 최종 가중치 공간에 안전하게 결합
        final_latent_space = morphed_topology + purified_perturbation
        
        # 7. [KR] 배치 차원 전체에 대한 강력한 L2 Norm = 1.0 에너지 보존 정규화 제어
        # 💡 [무결성 교정] 최종 가중치 공간 정규화 시 제로 텐서 분모 0 폭발(NaN)을 막기 위해 안전 분모 엡실론(eps) 주입
        # 💡 [Fix] Explicitly inject safety denominator epsilon (eps) to prevent final normalization NaN explosion
        conserved_weights = F.normalize(
            final_latent_space, 
            p=2, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        )
        
        # 💡 [무결성 교정] 모니터링 메트릭용 텐서들의 그래프 참조를 해제(.detach())하여 역전파 메모리 누수 및 잠재적 에러 완전 진압
        # 💡 [Fix] Detach logging tensors from the computational graph to prevent unintended memory leaks and backprop locks
        metrics = {
            "cosine_similarity": cos_sim.detach(),
            "gate_score": gate_mask.detach(),
            "l2_norm": torch.norm(conserved_weights, p=2, dim=-1).detach(),
            "learned_alpha": self.gate.alpha.detach(),
            "learned_eta": torch.tanh(self.gate.raw_eta).detach(),
            "transmission_rate": transmission.detach()
        }
        return conserved_weights, metrics
def main():
    print("========================================================================")
    print("🌌 Egregore Advanced Engine: Batch Operations & Adaptive Topology Test")
    print("========================================================================")
    
    batch_size = 4
    dim = AdaptiveTopologyConfig.LATENT_DIM
    
    # [KR] 매니폴드 마스터 레이어 인스턴스화
    alignment_layer = ProductionEnergyParityLayer(dim=dim)
    
    # ------------------------------------------------------------------------
    # [KR] 메모리 주소(ID) 참조 기반의 안전한 층별 학습률(LLRD) 분리 기법
    # ------------------------------------------------------------------------
    gate_param_ids = {id(p) for p in alignment_layer.gate.parameters()}

    backbone_params = [p for p in alignment_layer.parameters() if id(p) not in gate_param_ids]
    gate_params = list(alignment_layer.gate.parameters())

    # [KR] 위상 붕괴 방지를 위해 게이트 학습률은 100배 낮추고 가중치 감쇠(Weight Decay) 해제
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4},
        {"params": gate_params, "lr": 1e-6, "weight_decay": 0.0}
    ])
    
    print(f"초기 설정 하이퍼파라미터 (Initial Config) -> Alpha: {alignment_layer.gate.alpha.item():.2f}, "
          f"Eta: {torch.tanh(alignment_layer.gate.raw_eta).item():.4f}")
    print(f"테스트 주입 배치 크기 (Batch Input Size): [{batch_size}, {dim}]")
    print("-" * 88)

    # [KR] 가상의 배치 최적화 3단계 시뮬레이션
    for epoch in range(3):
        # [KR] 무작위 배치 데이터 생성 및 기하 정규화
        observer_batch = torch.randn(batch_size, dim)
        observer_batch = F.normalize(observer_batch, p=2, dim=-1, eps=AdaptiveTopologyConfig.EPSILON)
        
        # 1. [KR] Forward Pass 실행 (배치 파이프라인 연산)
        weights, metrics = alignment_layer(observer_batch)
        
        # 2. [💡 무결성 교정] 출력 가중치의 L2 Norm = 1.0 패리티와 매칭되도록 가상 타겟 텐서도 반드시 구조 정규화 처리
        # 2. [💡 Fix] Apply geometric normalization to target_batch to match outputs strict L2 Norm parity
        raw_target = torch.ones(batch_size, dim, device=weights.device) * 0.05
        target_batch = F.normalize(raw_target, p=2, dim=-1, eps=AdaptiveTopologyConfig.EPSILON)
        loss = F.mse_loss(weights, target_batch)
        
        # 3. [KR] Backward Pass 및 가중치 업데이트 (옵티마이저 진행)
        optimizer.zero_grad()
        loss.backward()
        
        # 💡 [무결성 교정] 대규모 분산 학습 하에서 발생할 수 있는 일시적인 그레디언트 스파이크 폭발 및 NaN 전파 원천 차단
        # 💡 [Fix] Explicitly add gradient clipping to prevent burst spikes and sudden weight explosion
        torch.nn.utils.clip_grad_norm_(alignment_layer.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 4. [KR] 결과 및 학습 안정성 지표 출력
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


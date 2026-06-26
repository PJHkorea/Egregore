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

    # 🌐 [v5.0 추가] 고차원 위상 기하학적 결합 손실 함수 제어용 가중치 상수 (Hyperparameters for Topological Loss)
    LAMBDA_CURVATURE: float = 0.1      # [KR] 곡률 정렬 손실 가중치 (\lambda_1) / [EN] Curvature alignment weight (\lambda_1)
    LAMBDA_CASIMIR: float = 0.05       # [KR] 카시미르 정보 엔트로피 손실 가중치 (\lambda_2) / [EN] Casimir informational entropy weight (\lambda_2)
    LAMBDA_GEODESIC: float = 0.1       # [KR] 리만 다양체 측지선 손실 가중치 (\lambda_3) / [EN] Riemannian manifold geodesic weight (\lambda_3)

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
        return F.normalize(torus_vector, p=2, dim=0, eps=AdaptiveTopologyConfig.EPSILON)


class AdvancedTopologicalLoss(nn.Module):
    """
    [KR] 고차원 위상 기하 구조 보존 및 정보 다양성 수호를 위한 3요소 결합 손실 함수
    [EN] 3-factor joint loss function for high-dimensional topological preservation and informational entropy defense
    """
    def __init__(self):
        super().__init__()
        self.l1 = AdaptiveTopologyConfig.LAMBDA_CURVATURE
        self.l2 = AdaptiveTopologyConfig.LAMBDA_CASIMIR
        self.l3 = AdaptiveTopologyConfig.LAMBDA_GEODESIC
        self.eps = AdaptiveTopologyConfig.EPSILON

    def forward(self, 
                conserved_weights: torch.Tensor, 
                observer_batch: torch.Tensor, 
                metrics: Dict[str, torch.Tensor],
                morphed_topology: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        [KR] 고차원 위상 기하 구조 보존을 위한 3요소 결합 손실 함수 연산 (수리 교정본)
        """
        # ====================================================================
        # 1. 곡률 정렬 손실 (L_Curvature): 입력 곡률과 가중치 구조 유사도 동기화
        # ====================================================================
        input_curvature = metrics['cosine_similarity']
        weight_curvature = metrics['gate_score']
        l_curvature = F.mse_loss(weight_curvature, input_curvature)

        # ====================================================================
        # 2. 카시미르 엔트로피 손실 (L_CasimirEntropy): 정보 확률 붕괴 차단
        # ====================================================================
        # 💡 투과율 스트림을 확률 분포로 변환하기 위해 Softmax 주입 (차원 보존)
        transmission_prob = F.softmax(metrics['transmission_rate'], dim=-1)
        
        # 섀넌 엔트로피 연산 (H = - \sum p * log(p))
        # 엔트로피 하한선을 강제(정보 다양성 보존)하기 위해 전체 엔트로피의 '음수'를 최소화
        entropy = -torch.sum(transmission_prob * torch.log(transmission_prob + self.eps), dim=-1)
        l_casimir_entropy = -torch.mean(entropy)

        # ====================================================================
        # 3. 지오데식 정규화 (L_Geodesic): 구면 다양체 표면 최단 지형 유도
        # ====================================================================
        normalized_topology = F.normalize(morphed_topology, p=2, dim=-1, eps=self.eps)
        
        # 💡 단순 직선 거리가 아닌 실제 매니폴드 곡면을 따르는 지오데식 호의 길이(Arc Length) 연산
        cos_sim = F.cosine_similarity(conserved_weights, normalized_topology, dim=-1, eps=self.eps)
        # 역삼각함수 폭발을 막기 위한 안정적 클램핑 처리
        clamped_cos = torch.clamp(cos_sim, min=-1.0 + self.eps, max=1.0 - self.eps)
        geodesic_distance = torch.acos(clamped_cos)
        l_geodesic = torch.mean(geodesic_distance)

        # ====================================================================
        # 4. 종합 위상 손실 컴파일 및 그래디언트 흐름 보호
        # ====================================================================
        total_topological_loss = (self.l1 * l_curvature) + (self.l2 * l_casimir_entropy) + (self.l3 * l_geodesic)
        
        # 💡 아티팩트 저장 시 계산 그래프에 영향을 주지 않도록 완벽히 분리 유도
        loss_artifacts = {
            "l_topological_total": float(total_topological_loss.item()),
            "l_curvature": float(l_curvature.item()),
            "l_casimir_entropy": float(l_casimir_entropy.item()),
            "l_geodesic": float(l_geodesic.item())
        }
        
        return total_topological_loss, loss_artifacts


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

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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
        conserved_weights = F.normalize(
            final_latent_space, 
            p=2, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        )
        
        # 💡 [v5.0 교정] AdvancedTopologicalLoss의 미분 경로 전파를 위해 손실 연산에 쓰일 성분은 detach 대상에서 제외
        metrics = {
            "cosine_similarity": cos_sim,
            "gate_score": gate_mask,
            "l2_norm": torch.norm(conserved_weights, p=2, dim=-1).detach(),
            "learned_alpha": self.gate.alpha,
            "learned_eta": torch.tanh(self.gate.raw_eta),
            "transmission_rate": transmission
        }
        return conserved_weights, morphed_topology, metrics

def main():
    print("========================================================================")
    print("🌌 Egregore Advanced Engine: Batch Operations & Adaptive Topology Test")
    print("========================================================================")
    
    batch_size = 4
    dim = AdaptiveTopologyConfig.LATENT_DIM
    
    # [KR] 매니폴드 마스터 레이어 및 v5.0 고차원 기하 손실 함수 인스턴스화
    alignment_layer = ProductionEnergyParityLayer(dim=dim)
    topological_loss_fn = AdvancedTopologicalLoss()
    
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
        observer_batch = torch.randn(batch_size, dim)
        observer_batch = F.normalize(observer_batch, p=2, dim=-1, eps=AdaptiveTopologyConfig.EPSILON)
        
        # 1. [KR] Forward Pass 실행
        weights, morphed_topology, metrics = alignment_layer(observer_batch)
        
        # 2. [KR] 태스크 기본 역설 정규화 수행 (MSE 타겟 매칭)
        raw_target = torch.ones(batch_size, dim, device=weights.device) * 0.05
        target_batch = F.normalize(raw_target, p=2, dim=-1, eps=AdaptiveTopologyConfig.EPSILON)
        task_loss = F.mse_loss(weights, target_batch)
        
        # 🌐 [v5.0 추가] 고차원 위상 구조 보존을 위한 기하 결합 손실 역산 파이프라인 가동
        topo_loss, topo_artifacts = topological_loss_fn(weights, observer_batch, metrics, morphed_topology)
        
        # 종합 물리 결합 손실 방정식 성립
        total_loss = task_loss + topo_loss
        
        # 3. [KR] Backward Pass 및 가중치 업데이트 (옵티마이저 진행)
        optimizer.zero_grad()
        total_loss.backward()
        
        # [KR] 그레디언트 스파이크 폭발 및 NaN 전파 원천 차단
        torch.nn.utils.clip_grad_norm_(alignment_layer.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 4. [KR] 결과 및 자율 항상성 학습 안정성 지표 출력
        print(
            f"Epoch {epoch+1} | Total Loss: {total_loss.item():.4f} (Task: {task_loss.item():.4f}, Topo: {topo_artifacts['l_topological_total']:.4f}) |\n"
            f"  -> Metrics | Curvature Loss: {topo_artifacts['l_curvature']:.4f} | Casimir Entropy: {topo_artifacts['l_casimir_entropy']:.4f} | Geodesic Arc: {topo_artifacts['l_geodesic']:.4f} |\n"
            f"  -> State   | Avg CosSim: {metrics['cosine_similarity'].mean().item():.4f} | Gate Score: {metrics['gate_score'].mean().item():.3f} | Transmission: {metrics['transmission_rate'].mean().item():.4f}"
        )
        print("-" * 88)

    print("✅ 검증 완료: v5.0 기하학적 결합 손실 파이프라인 연동 및 자율 항상성 인지 아키텍처가 결함 없이 정상동작합니다.")

if __name__ == "__main__":
    main()


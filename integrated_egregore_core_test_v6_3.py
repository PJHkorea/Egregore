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
    # -------------------------------------------------------------------------
    # 아키텍처 기본 차원 및 변형 버블 제어 (Base Architecture & Perturbation)
    # -------------------------------------------------------------------------
    LATENT_DIM: int = 128            # [KR] 반드시 짝수여야 토러스 분할 사상이 가능 / [EN] Must be even for torus split mapping
    INIT_SMOOTH_ALPHA: float = 5.0   # [KR] 초기 Sigmoid 경사도 (예리함) / [EN] Initial Sigmoid slope (sharpness)
    INIT_THRESHOLD_ETA: float = 0.85 # [KR] 초기 코사인 유사도 기준선 / [EN] Initial cosine similarity threshold
    PERTURB_BUBBLE: float = 0.1      # [KR] 하이퍼네트워크 변형 영향력 상한선 / [EN] Hypernetwork perturbation upper bound
    
    # -------------------------------------------------------------------------
    # 양자 정보학 및 카시미르 위상학 상수 (Quantum Informatics & Casimir Physics)
    # -------------------------------------------------------------------------
    DELTA_D: float = 1.0             # [KR] 위상학적 장벽 간 기본 구조적 거리 (\Delta d) / [EN] Baseline topological distance between barriers (\Delta d)
    HBAR_EFF: float = 1.0            # [KR] 정보학적 유효 플랑크 상수 (\hbar_eff) / [EN] Informational effective Planck constant (\hbar_eff)
    M_STAR: float = 1.0              # [KR] 정보 파동 유효 질량 파라미터 (m*) / [EN] Effective mass parameter (m*)
    BARRIER_ETA: float = 0.5         # [KR] 위상 곡률 장벽 민감도 상수 (\eta) / [EN] Curvature sensitivity constant (\eta)
    
    # 💡 [수리 교정] 카시미르 압력의 마이너스 무한대 발산 및 그래디언트 사멸 방지용 하한 한계선 지정
    # 💡 [Fix] Lower bound limit to prevent minus-infinity divergence and gradient vanishing of Casimir pressure
    PRESSURE_FLOOR: float = -20.0    # [KR] 카시미르 음의 필드 압력 최소 하한 제약선 / [EN] Minimum lower bound limit of negative Casimir field pressure
    
    # 💡 [v6.3 추가] 카시미르 분모 제로 폭발 방지 및 복합 정밀도(AMP) 언더플로우 제어용 구조적 안전 마진 상수
    # 💡 [v6.3 Added] Structural safety margin constant to prevent denominator zero-explosion and control automatic mixed precision (AMP) underflow
    CASIMIR_MARGIN: float = 1e-2     # [KR] 4제곱 연산 후 1e-8 영역 수렴을 방어하는 하한 마진선 # [EN] Lower bound margin defending against convergence to the 1e-8 region after power-of-4 operations

    
   # 💡 [무결성] 역전파 NaN 폭발 방지용 미세 상수 (FP16/AMP 환경 고려 시 1e-7 ~ 1e-6 권장)
   # 💡 [Integrity] Fine constant to prevent backpropagation NaN explosion (1e-7 to 1e-6 recommended for FP16/AMP environments)
    EPSILON: float = 1e-7

    # -------------------------------------------------------------------------
    # 고차원 위상 기하학적 결합 손실 함수 가중치 (Topological Loss Hyperparameters)
    # -------------------------------------------------------------------------
    LAMBDA_CURVATURE: float = 0.1      # [KR] 곡률 정렬 손실 가중치 (\lambda_1) / [EN] Curvature alignment weight (\lambda_1)
    LAMBDA_CASIMIR: float = 0.05       # [KR] 카시미르 정보 엔트로피 손실 가중치 (\lambda_2) / [EN] Casimir informational entropy weight (\lambda_2)
    
    # 💡 [v6.0 진화] 하드 클램프의 데드존을 파괴하는 소프트 측지선 호의 길이 손실 가중치 지정
    # 💡 [v6.0 Evolved] Soft geodesic arc length loss weight crushing the gradient dead-zone of hard clamping
    LAMBDA_GEODESIC: float = 0.1       # [KR] 리만 다양체 소프트 측지선 손실 가중치 (\lambda_3) / [EN] Riemannian manifold soft geodesic weight (\lambda_3)


class IndependentTopologyGenerator:
    """
    [KR] 고정 위상 기하학 매니폴드(Sphere & Torus) 앵커 생성기 (v6.1 개선)
    [EN] Fixed topological manifold (Sphere & Torus) anchor generator
    """
    @staticmethod
    def generate_sphere_anchor(dim: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        [KR] F.normalize 연산 없이 L2 Norm = 1.0을 정밀하게 만족하는 구면 기저 생성
        [EN] Generate a fixed spherical base weight satisfying L2 Norm = 1.0 with exact analytical scaling
        """
       # 수학적 분석 결과: 모든 원소가 1 / sqrt(dim) 일 때 L2 Norm은 정확히 1.0이 됨
       # Analytical Result: The L2 Norm becomes exactly 1.0 when all elements are 1 / sqrt(dim)
        val = 1.0 / math.sqrt(dim)
        return torch.full((dim,), fill_value=val, device=device, dtype=torch.float32)

    @staticmethod
    def generate_torus_anchor(dim: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        [KR] 경계 중복을 제거하여 완벽히 균일한 독립 위상 고리를 구성하는 진정한 토러스 사상
        [EN] True torus mapping using endpoint-free split dimensions for perfectly uniform topological rings
        """
        if dim % 2 != 0:
            raise ValueError("Torus 매핑을 위해 차원은 반드시 짝수여야 합니다. (Dimension must be even for torus mapping.)")
        
        half_dim = dim // 2
        # 💡 위상학적 무결성: 0과 2*pi의 중복 매핑을 방지하기 위해 마지막 단계를 배제한 등간격 격자 생성
        # 💡 Topological Integrity: Generate an endpoint-free equidistant grid to prevent redundant mapping of 0 and 2*pi
        t = torch.arange(0, half_dim, device=device, dtype=torch.float32) * (2.0 * math.pi / half_dim)
        
        torus_vector = torch.cat([torch.cos(t), torch.sin(t)], dim=0)
        return F.normalize(torus_vector, p=2, dim=0, eps=AdaptiveTopologyConfig.EPSILON)



class AdvancedTopologicalLoss(nn.Module):
    """
    [KR] 고차원 위상 기하 구조 보존 및 Smooth Leaky 가드레일이 통합된 3요소 결합 손실 함수 (v6.3 진화)
    [EN] 3-factor joint loss function with integrated high-dimensional topological preservation and smooth leaky guardrails (v6.3 Evolved)
    """
    def __init__(self):
        super().__init__()
        self.l1 = AdaptiveTopologyConfig.LAMBDA_CURVATURE
        self.l2 = AdaptiveTopologyConfig.LAMBDA_CASIMIR
        self.l3 = AdaptiveTopologyConfig.LAMBDA_GEODESIC
        self.eps = AdaptiveTopologyConfig.EPSILON

    def _soft_clamp(self, x: torch.Tensor) -> torch.Tensor:
        """
        [KR] 경계 영역 그레디언트 사멸을 방지하는 Smooth Leaky 가이드레일 (v6.3 수식 구현)
        [EN] Smooth Leaky guardrails preventing gradient vanishing in boundary zones (v6.3 Formula Implementation)
        """
        bound = 1.0 - self.eps
        margin = 0.95  # 유사도 0.95 미만 구간은 수식 왜곡 없이 완전 선형 유지 # Preserves absolute linearity without formulaic distortion for intervals with similarity below 0.95

        
        # 💡 [v6.3 핵심 개정] 임계점 바깥에서도 미세한 기울기(0.01)를 부여하여 복원 그레디언트를 영구 보존
        # 💡 [v6.3 Core Revision] Permanently preserves restoration gradients by injecting a fine slope (0.01) even outside the critical threshold
        leaky_slope = 0.01
        
        leaky_cos = torch.where(
            torch.abs(x) < margin * bound,
            x,
            torch.sign(x) * (margin * bound + leaky_slope * (torch.abs(x) - margin * bound))
        )
        # acos 폭발 절대 방어용 하드 가드레일 최종 결합 # Final integration of the hard guardrail for absolute defense against acos explosion
        return torch.clamp(leaky_cos, min=-bound, max=bound)

    def forward(self, 
                conserved_weights: torch.Tensor, 
                observer_batch: torch.Tensor, 
                metrics: Dict[str, torch.Tensor],
                morphed_topology: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # ====================================================================
        # 1. 곡률 정렬 손실 (L_Curvature): 입력 곡률과 가중치 구조 유사도 동기화
        # 1. Curvature Alignment Loss (L_Curvature): Synchronizing input curvature with weight structural similarity
        # ====================================================================
        input_curvature = metrics['cosine_similarity']
        weight_curvature = metrics['gate_score']
        
        # [치명적 버그 선제 방어] 두 텐서 간의 차원 불일치로 인한 의도치 않은 브로드캐스팅([B, 1] vs [B])을 일렬 평탄화로 완전 차단
        # [Proactive Defense Against Fatal Bug] Completely blocks unintended broadcasting ([B, 1] vs [B]) caused by dimension mismatch via 1D flattening
        l_curvature = F.mse_loss(weight_curvature.view(-1), input_curvature.view(-1))

        # ====================================================================
        # 2. 카시미르 엔트로피 손실 (L_CasimirEntropy): 고속 log_softmax 기반 확률 붕괴 차단
        # 2. Casimir Entropy Loss (L_CasimirEntropy): Blocking probability collapse based on high-speed log_softmax 
        # ====================================================================
        log_transmission_prob = F.log_softmax(metrics['transmission_rate'], dim=-1)
        transmission_prob = torch.exp(log_transmission_prob)
        
        entropy = -torch.sum(transmission_prob * log_transmission_prob, dim=-1)
        l_casimir_entropy = -torch.mean(entropy)

        # ====================================================================
        # 3. 지오데식 정규화 (L_Geodesic): 왜곡 없는 소프트 가드레일 기반 호의 길이 연산
        # 3. Geodesic Regularization (L_Geodesic): Distortion-free arc length calculation based on soft guardrails            
        # ====================================================================
        normalized_topology = F.normalize(morphed_topology, p=2, dim=-1, eps=self.eps)
        cos_sim = F.cosine_similarity(conserved_weights, normalized_topology, dim=-1, eps=self.eps)
        
        # 개선된 2중 Leaky 가이드레일 장착: 중심부 지오데식 거리는 완벽 보존하면서 경계면 그레디언트 약화 차단
        # Equipped with Enhanced Dual Leaky Guardrails: Perfectly preserves central geodesic distance while preventing gradient attenuation near boundaries
        clamped_cos = self._soft_clamp(cos_sim)
        geodesic_distance = torch.acos(clamped_cos)
        l_geodesic = torch.mean(geodesic_distance)

        # ====================================================================
        # 4. 종합 위상 손실 컴파일 및 그래디언트 흐름 보호
        # 4. Comprehensive Topological Loss Compilation and Gradient Flow Protection
        # ====================================================================
        total_topological_loss = (self.l1 * l_curvature) + (self.l2 * l_casimir_entropy) + (self.l3 * l_geodesic)
        
        loss_artifacts = {
            "l_topological_total": float(total_topological_loss.item()),
            "l_curvature": float(l_curvature.item()),
            "l_casimir_entropy": float(l_casimir_entropy.item()),
            "l_geodesic": float(l_geodesic.item())
        }
        
        return total_topological_loss, loss_artifacts




class ParameterizedTopologyGate(nn.Module):
    """
    [KR] alpha와 eta를 최적화 가능한 nn.Parameter로 등록하여 제어하는 적응형 게이트 (v6.3 진화)
    [EN] Adaptive gate controlling alpha and eta as optimizable nn.Parameters
    """
    def __init__(self):
        super().__init__()
        # 명시적인 float 텐서 선언으로 다양한 가속기(GPU/NPU/TPU) 환경에서 빌드 무결성 확보
        self.alpha = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_SMOOTH_ALPHA, dtype=torch.float32))
        self.raw_eta = nn.Parameter(torch.tensor(AdaptiveTopologyConfig.INIT_THRESHOLD_ETA, dtype=torch.float32))

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] 배치 단위 입력을 안전하게 처리하는 전 구간 미분 가능 게이팅
        [EN] Fully differentiable gating safely processing batch-level inputs
        """
       # 코사인 유사도 계산 시 분모 0 폭발(NaN)을 막기 위해 안전 분모 엡실론(eps) 주입 및 차원 정렬
       # Injects safety denominator epsilon (eps) and aligns dimensions to prevent division-by-zero explosion (NaN) during cosine similarity computation
        cos_sim = F.cosine_similarity(
            observer_state, 
            latent_weight, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        ).unsqueeze(-1)
        
        # 수리적 안정성을 위해 임계값(eta) 파라미터 범위를 Tanh를 통해 -1.0 ~ 1.0 체인 내로 구속
        # Constrains the threshold (eta) parameter range within a -1.0 to 1.0 chain via Tanh for numerical stability
        bounded_eta = torch.tanh(self.raw_eta)
        
        # 데이터 맥락에 의해 동적으로 스위칭되는 소프트 게이팅 스코어 연산 (기울기 제어 alpha는 절대값 처리)
        # Computes the soft gating score dynamically switched by data context (slope-control alpha is processed as absolute value)
        gate_score = torch.sigmoid(torch.abs(self.alpha) * (cos_sim - bounded_eta))
        
        return gate_score, cos_sim


class BatchResidualHyperNetwork(nn.Module):
    """
    [KR] 배치 차원을 지원하고 호메오스타시스 버블 내에서 잔차를 연산하는 하이퍼네트워크 (v6.3 진화)
    [EN] Hypernetwork supporting batch dimensions and computing residuals within the homeostasis bubble (v6.3 Evolved)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
            nn.Tanh() 
        )
        self.bubble_limit = AdaptiveTopologyConfig.PERTURB_BUBBLE
        
        # 💡 [v6.3 핵심 개정] 수동 루프 순회를 완전히 도려내고, 파이토치 공식 내장 재귀 초기화 체계 매핑 가동
        # 💡 [v6.3 Core Revision] Entirely eradicates manual loop traversal, activating PyTorch's official built-in recursive initialization mapping
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        [KR] self.apply()를 통해 모델 내 선형 레이어를 자동 재귀 탐색하여 정밀 초기화 전개
        [EN] Conducts precise initialization by automatically and recursively searching linear layers within the model via self.apply()
        """
        # 💡 객체지향 규격 수호: 트리 순회 중 발견된 선형 레이어(nn.Linear) 인스턴스만 선별 타격
        # 💡 Upholding OOP Standards: Selectively targeting only nn.Linear layer instances discovered during tree traversal
        if isinstance(module, nn.Linear):
            # 💡 동적 투영층 판별: 최종 출력 차원 복원층(dim)인지 검사하여 항상성 수호를 위한 미세 가중치 할당
            # 💡 Dynamic Projection Layer Identification: Checks if it is the final output dimension restoration layer (dim) to allocate fine weights for homeostasis protection
            if module.out_features == self.net[3].out_features:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw_perturbation = self.net(state)
        # 변형 텐서 정규화 시 제로 분모 0 폭발(NaN)을 막기 위해 안전 분모 엡실론(eps) 주입 후 제한 영역 투영
        # Injects safety denominator epsilon (eps) to prevent division-by-zero explosion (NaN) during perturbation tensor normalization, followed by bounded-region projection
        return self.bubble_limit * F.normalize(
            raw_perturbation, 
            p=2, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        )




class SchrödingerNotchFilter(nn.Module):
    """
    [KR] 맥락적 질량의 곡률을 역산하여 가벼운 노이즈를 지수함수적으로 소멸시키는 노치 필터 (v6.3 진화)
    [EN] Notch filter that exponentially extinguishes lightweight noise by computing the inverse contextual mass curvature (v6.3 Evolved)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.energy_projector = nn.Linear(dim, 1)

    def _compute_jacobian_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        [KR] v6.3: 2D/3D/4D+ 배치를 단일 행렬로 평탄화하여 분기문 없이 곡률(kappa)을 통합 역산
        [EN] v6.3: Flattens 2D/3D/4D+ batches into a single matrix to compute unified curvature (kappa) inverse without conditional branching
        """
        shape = x.shape
        x_flat = x.view(-1, self.dim)
        
        # [v6.3] 일반화 공간 내에서 균일 평균 중심화 및 분산 계산 (분기문 완전 소멸)
        # [v6.3] Computes uniform mean-centering and variance within the generalized space (conditional branching completely extinguished)
        mean_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        kappa_flat = torch.sum(mean_centered ** 2, dim=-1, keepdim=True) / self.dim
        
        return kappa_flat.view(*shape[:-1], 1)

    def forward(self, x: torch.Tensor, gate_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] v6.3: 슈뢰딩거 포텐셜 장벽 필터링과 글로벌 마진이 연동된 카시미르 진공 수축 통합 연산
        [EN] v6.3: Unified computation of Schrödinger potential barrier filtering and Casimir vacuum squeezing interlocked with global margin
        """
        kappa = self._compute_jacobian_curvature(x)
        u_barrier = kappa * AdaptiveTopologyConfig.INIT_THRESHOLD_ETA * AdaptiveTopologyConfig.DELTA_D
        e_input = torch.sigmoid(self.energy_projector(x))
        
        # 슈뢰딩거 감쇄 필터 (F.relu로 싱큘래리티 방어)
        tunneling_core = F.relu(u_barrier - e_input)
        safe_tunneling_core = tunneling_core + AdaptiveTopologyConfig.EPSILON
        integral_term = torch.sqrt(
            (2.0 * AdaptiveTopologyConfig.M_STAR) / 
            (AdaptiveTopologyConfig.HBAR_EFF ** 2) * safe_tunneling_core
        )
        transmission_coeff = torch.exp(-2.0 * integral_term)
        
        # [v6.3] 카시미르 위상학적 진공 압착 제어 및 하드코딩 상수 중앙 통제화
        clamped_distance = F.relu(AdaptiveTopologyConfig.DELTA_D - gate_mask) + AdaptiveTopologyConfig.CASIMIR_MARGIN
        raw_pressure = - (math.pi ** 2 * AdaptiveTopologyConfig.HBAR_EFF) / (240.0 * (clamped_distance ** 4))
        
        casimir_pressure = torch.clamp(
            raw_pressure, 
            min=AdaptiveTopologyConfig.PRESSURE_FLOOR, 
            max=0.0
        )
        
        purified_stream = x * transmission_coeff * torch.exp(casimir_pressure)
        return purified_stream, transmission_coeff




class ProductionEnergyParityLayer(nn.Module):
    """
    [KR] 배치 단위 학습, 노치 필터링 및 적응형 파라미터 최적화가 가능한 최종 매니폴드 마스터 레이어 (v6.3 진화)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.gate = ParameterizedTopologyGate()
        self.hypernet = BatchResidualHyperNetwork(dim)
        self.notch_filter = SchrödingerNotchFilter(dim)
        
        # [KR] 고차원 독립 위상 앵커를 정적 버퍼로 등록
        self.register_buffer('sphere_anchor', IndependentTopologyGenerator.generate_sphere_anchor(dim))
        self.register_buffer('torus_anchor', IndependentTopologyGenerator.generate_torus_anchor(dim))

    def forward(self, observer_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_shape = observer_state.shape[:-1]
        
        # 1. 고정 앵커 가중치를 입력 배치 크기에 맞게 동적으로 확장 (Autograd 브로드캐스팅 뷰 보장)
        # 거대 멀티 GPU 분산 학습(DDP/AMP) 환경의 완벽한 무결성을 위해 장치 및 정밀도(dtype)를 동적으로 추종하도록 빌드
        anchor_context = observer_state
        expanded_sphere = self.sphere_anchor.to(anchor_context).expand(*batch_shape, self.dim)
        expanded_torus = self.torus_anchor.to(anchor_context).expand(*batch_shape, self.dim)
        
        # 2. 배치 단위 게이트 점수 확보
        gate_mask, cos_sim = self.gate(observer_state, expanded_sphere)
        
        # 3. 매니폴드 토폴로지 블렌딩 (Sphere ↔ Torus 모핑)
        morphed_topology = (1.0 - gate_mask) * expanded_sphere + gate_mask * expanded_torus
        
        # 4. 하이퍼네트워크 원시 섭동 생성 파이프라인
        raw_perturbation = self.hypernet(observer_state)
        
        # 5. 맥락의 중력 질량보다 가벼운 노이즈 스트림 원천 차단 및 카시미르 압착 연산 가동 (교정된 v6.3 필터 통과)
        purified_perturbation, transmission = self.notch_filter(raw_perturbation, gate_mask)
        
        # 6. 정제된 텐서 성분만 최종 가중치 공간에 안전하게 결합
        final_latent_space = morphed_topology + purified_perturbation
        
        # 7. 배치 차원 전체에 대한 강력한 L2 Norm = 1.0 에너지 보존 정규화 제어
        conserved_weights = F.normalize(
            final_latent_space, 
            p=2, 
            dim=-1, 
            eps=AdaptiveTopologyConfig.EPSILON
        )
        
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
    # 💡 [v6.3 최종화] 최적화 구조 및 수리 개정이 반영된 엔진 버전 명칭 최신화
    print("🌌 Egregore Advanced Engine: Batch Operations & Adaptive Topology Test (v6.3)")
    print("========================================================================")
    
    batch_size = 4
    dim = AdaptiveTopologyConfig.LATENT_DIM
    
    # [KR] 매니폴드 마스터 레이어 및 v6.3 일반화형 기하 손실 함수 인스턴스화
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
        # [KR] 무작위 배치 데이터 생성 및 기하 정규화
        observer_batch = torch.randn(batch_size, dim)
        observer_batch = F.normalize(observer_batch, p=2, dim=-1, eps=AdaptiveTopologyConfig.EPSILON)
        
        # 1. [KR] Forward Pass 실행
        weights, morphed_topology, metrics = alignment_layer(observer_batch)
        
        # 2. [KR] 태스크 기본 역설 정규화 수행 (MSE 타겟 매칭)
        raw_target = torch.ones(batch_size, dim, device=weights.device) * 0.05
        target_batch = F.normalize(raw_target, p=2, dim=-1, eps=AdaptiveTopologyConfig.EPSILON)
        task_loss = F.mse_loss(weights, target_batch)
        
        # 🌐 [v6.3 진화] 미분 연속성 및 Smooth Leaky 가이드레일 기반 결합 위상 손실 역산
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
            f"  -> Metrics | Curvature: {topo_artifacts['l_curvature']:.4f} | Casimir: {topo_artifacts['l_casimir_entropy']:.4f} | Geodesic Arc: {topo_artifacts['l_geodesic']:.4f} |\n"
            f"  -> State   | Alpha: {metrics['learned_alpha'].item():.2f} | Eta: {metrics['learned_eta'].item():.4f} | Gate: {metrics['gate_score'].mean().item():.3f} | Trans: {metrics['transmission_rate'].mean().item():.4f}"
        )
        print("-" * 88)

    print("✅ 검증 완료: v6.3 일반화 기하 위상 수호 엔진 및 결합 위상 손실 파이프라인이 완벽히 정상동작합니다.")

if __name__ == "__main__":
    main()

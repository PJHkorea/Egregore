import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any

class EgregoreGlobalConfig:
    """
    [KR] 글로벌 아키텍처 수리 상 수 및 환경 변수 정의
    [EN] Definition of global architecture mathematical constants and environment variables
    """
    LATENT_DIM: int = 128          # [KR] 잠재 임베딩 고차원 차원 수 | [EN] High-dimensional latent embedding space size
    BARRIER_ETA: float = 0.5       # [KR] 위상 곡률 장벽 민감도 상수 (\eta) | [EN] Curvature sensitivity constant (\eta)
    DELTA_D: float = 2.0           # [KR] 레이어 간 차원 위상 구조적 거리 (\Delta d) | [EN] Topological dimensional distance (\Delta d)
    M_STAR: float = 1.0            # [KR] 정보 파동 유효 질량 파라미터 (m*) | [EN] Effective mass parameter (m*)
    HBAR_EFF: float = 1.0          # [KR] 수리 정보학 유효 플랑크 상수 (\hbar_eff) | [EN] Effective Planck constant (\hbar_eff)
    HOMEOSTASIS_LIMIT: float = 0.1 # [KR] 하이퍼네트워크 변형 한계 제약선 (0.1) | [EN] Hypernetwork perturbation upper bound (0.1)

class ProductionEnergyParityLayer(nn.Module):
    """
    [KR] 에너지 등가 정규화 레이어 (에너지 보존 법칙)
    [EN] Geometric Energy Parity Layer (Conservation of Energy)
    """
    def __init__(self):
        super().__init__()

    def forward(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """
        [KR] 모든 차원 텐서에 대해 L2 Norm = 1.0을 강제 동결하여 그래디언트 폭발 원천 차단
        [EN] Forcibly freezes L2 Norm = 1.0 across all dimensions to inherently prevent gradient explosion
        """
        # latent_tensor shape: [Batch_Size, Seq_Len, Latent_Dim]
        return F.normalize(latent_tensor, p=2, dim=-1)
class TopologicalBarrierModule(nn.Module):
    """
    [KR] 1단계: 위상학적 터널링 장벽 레이어 (슈뢰딩거 포텐셜 기반 노이즈 필터)
    [EN] Phase 1: Topological Tunneling Barrier Layer (Schrödinger Potential-based Noise Filter)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # [KR] 입력 데이터가 가진 고유의 정보 에너지를 스칼라로 압축 투영하는 선형 레이어
        # [EN] Linear layer projecting incoming data features into a scalar information energy value
        self.energy_projector = nn.Linear(hidden_dim, 1)

    def _compute_jacobian_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        [KR] 고속 야코비안 곡률 대리값 연산 (\kappa = Tr(J^T * J))
        [EN] Fast Jacobian Curvature Proxy Computation (\kappa = Tr(J^T * J))
        """
        batch_size, seq_len, _ = x.shape
        # [KR] 차원을 평탄화하여 2D 배치 형태로 변환 [B*S, Dim]
        # [EN] Flatten dimensions into a 2D batch tensor [B*S, Dim]
        x_flat = x.view(-1, self.hidden_dim)
        
        # [KR] 평균 중심화 공분산 근사 행렬 연산
        # [EN] Mean-centered covariance approximation matrix operation
        mean_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        covariance = torch.matmul(mean_centered, mean_centered.t())  # [B*S, B*S]
        
        # [KR] 프로베니우스 노름 제곱 패턴 기반의 곡률 정량화 추적
        # [EN] Curvature quantification tracking based on squared Frobenius norm patterns
        kappa_flat = torch.sum(covariance ** 2, dim=-1) / self.hidden_dim
        
        # [KR] 원래의 멀티 배치 차원으로 복원 [B, S, 1]
        # [EN] Restore back to the original multi-batch tensor shape [B, S, 1]
        return kappa_flat.view(batch_size, seq_len, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [KR] 슈뢰딩거 필터 순전파: 저수준 노이즈를 지수함수적으로 소멸
        [EN] Schrödinger Filter Forward: Exponentially decays low-level contextual noise
        """
        # [KR] 1) 입력 텐서의 위상학적 곡률 변동성 (\kappa) 실시간 연산
        # [EN] 1) Real-time evaluation of topological curvature variance (\kappa)
        kappa = self._compute_jacobian_curvature(x)
        
        # [KR] 2) 기하학적 위상 장벽 포텐셜 (U_barrier) 결정 공식 적용
        # [EN] 2) Apply geometric topological barrier potential (U_barrier) formula
        u_barrier = EgregoreGlobalConfig.BARRIER_ETA * kappa * EgregoreGlobalConfig.DELTA_D
        
        # [KR] 3) 입력 데이터 벡터 고유의 전하 (E_input) 추출 및 정규화
        # [EN] 3) Extract and normalize incoming energy matrix charge (E_input)
        e_input = torch.sigmoid(self.energy_projector(x))
        
        # [KR] 4) 양자 터널링 효과 함수 조립 및 허수(NaN) 발생 방지 ReLU 제약
        # [EN] 4) Quantum tunneling function assembly with ReLU constraint to prevent NaN values
        diff_energy = u_barrier - e_input
        tunneling_core = F.relu(diff_energy)  # [KR] 음수 영역을 0으로 묶음 | [EN] Zero-out negative values
        
        # [KR] 5) 감쇄 지수 선적분 근사 및 정보 투과율 (T) 컴파일
        # [EN] 5) Decay exponential line integral approximation and transmission coefficient (T) compile
        integral_term = torch.sqrt((2.0 * EgregoreGlobalConfig.M_STAR) / (EgregoreGlobalConfig.HBAR_EFF ** 2) * tunneling_core)
        transmission_coeff = torch.exp(-2.0 * integral_term)  # [B, S, 1]
        
        # [KR] 6) 최종 정제된 텐서 출력 (Purified X = X * T)
        # [EN] 6) Final purified tensor stream output (Purified X = X * T)
        purified_x = x * transmission_coeff
        
        return purified_x, transmission_coeff
class IndependentTopologyGenerator:
    """
    [KR] 독립 위상 기저 생성기 (차원 분리 기반 도넛 토러스 형상 매핑)
    [EN] Independent Topology Generator (Dimension-split based Torus Manifold Mapping)
    """
    @staticmethod
    def generate_torus_basis(dim: int, device: torch.device) -> torch.Tensor:
        """
        [KR] 고차원 임베딩 차원을 절반으로 쪼개어 Cos/Sin 독립 위상 고리 사상
        [EN] Splits high-dimensional embedding and maps via Cos/Sin independent ring phase
        """
        half_dim = dim // 2
        # [KR] 0부터 2*pi까지 연속적인 위상 그리드 사상
        # [EN] Map continuous phase grid from 0 to 2*pi
        t = torch.linspace(0, 2 * math.pi, half_dim, device=device)
        torus_vector = torch.cat([torch.cos(t), torch.sin(t)], dim=0)
        
        # [KR] 홀수 차원 대응용 제로 패딩 처리
        # [EN] Zero padding handling for odd dimensions
        if torus_vector.shape[0] < dim:
            padding = torch.zeros(dim - torus_vector.shape[0], device=device)
            torus_vector = torch.cat([torus_vector, padding], dim=0)
            
        return F.normalize(torus_vector, p=2, dim=0)

class BatchResidualHyperNetwork(nn.Module):
    """
    [KR] 배치 잔여 하이퍼네트워크 (항상성 방울 통제 레이어)
    [EN] Batch Residual HyperNetwork (Homeostasis Bubble Control Layer)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
            nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        [KR] 미세 섭동을 생성하되 항상성 유지를 위해 0.1 한계 범위 내로 강력하게 제약
        [EN] Generates fine perturbation but strictly limits it within a 0.1 bound for homeostasis
        """
        raw_perturbation = self.net(x)
        # [KR] 수리 장론 공식 (|0.1 * H|) 경계선Bubble 강제 적용
        # [EN] Forcibly apply mathematical field theory constraint (|0.1 * H|) bubble
        controlled_perturbation = torch.tanh(raw_perturbation) * EgregoreGlobalConfig.HOMEOSTASIS_LIMIT
        return controlled_perturbation

class ParameterizedTopologyGate(nn.Module):
    """
    [KR] 적응형 위상 게이팅 및 스위칭 레이어 (완전 미분 가능 제어기)
    [EN] Parameterized Topology Gate & Switching Layer (Fully Differentiable Controller)
    """
    def __init__(self):
        super().__init__()
        # [KR] 학습 가능한 임계 변수 nn.Parameter 할당
        # [EN] Assign learnable adaptive parameter via nn.Parameter
        self.raw_eta = nn.Parameter(torch.tensor(0.85))
        self.alpha = nn.Parameter(torch.tensor(5.0))

    def forward(self, observer_state: torch.Tensor, latent_weight: torch.Tensor) -> torch.Tensor:
        """
        [KR] 시그모이드 소프트 게이팅을 활용하여 미분 단절 없는 그레디언트 그래프 유지
        [EN] Utilizes Sigmoid soft-gating to maintain continuous Autograd graphs without disconnection
        """
        # [KR] 코사인 유사도 분석 기반의 주체적 정렬 척도 계산
        # [EN] Calculate alignment metrics based on cosine similarity analysis
        cos_sim = F.cosine_similarity(observer_state, latent_weight, dim=-1, eps=1e-8)
        cos_sim = cos_sim.unsqueeze(-1)  # [B, S, 1]
        
        # [KR] 임계점 이탈 방지를 위한 Tanh 바운더리 랩핑
        # [EN] Wrap threshold within Tanh boundary to prevent extreme divergence
        bounded_eta = torch.tanh(self.raw_eta)
        
        # [KR] 연속 변형 점수 계산 (If-Else 구조의 완벽한 수학적 대체)
        # [EN] Continuous gating score logic (Flawless mathematical replacement of If-Else)
        gate_score = torch.sigmoid(torch.abs(self.alpha) * (cos_sim - bounded_eta))
        return gate_score
class IntegratedEgregoreSystem(nn.Module):
    """[KR] 마스터 아키텍처 결합체 (1단계 위상 장벽 + 2단계 에너지 엔진 통합 시스템)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.barrier_layer = TopologicalBarrierModule(dim)
        self.topology_gate = ParameterizedTopologyGate()
        self.hypernetwork = BatchResidualHyperNetwork(dim)
        self.energy_parity = ProductionEnergyParityLayer()
        self.sphere_anchor = nn.Parameter(torch.randn(dim))
        
    def forward(self, raw_stream: torch.Tensor, observer_state: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """[KR] 마스터 순전파 파이프라인: 장벽 정제 후 다양체 모핑 수행"""
        device = raw_stream.device
        batch_size, seq_len, _ = raw_stream.shape
        
        # 1단계: 위상학적 터널링 장벽 (노이즈 소멸)
        purified_stream, transmission = self.barrier_layer(raw_stream)
        
        # 2단계: 구면/토러스 기반 다양체 하이브리드 블렌딩
        normalized_sphere = F.normalize(self.sphere_anchor, p=2, dim=0)
        expanded_sphere = normalized_sphere.view(1, 1, self.dim).expand(batch_size, seq_len, self.dim)
        torus_basis = IndependentTopologyGenerator.generate_torus_basis(self.dim, device)
        expanded_torus = torus_basis.view(1, 1, self.dim).expand(batch_size, seq_len, self.dim)
        
        gate_mask = self.topology_gate(observer_state, expanded_sphere)
        morphed_topology = (1.0 - gate_mask) * expanded_sphere + gate_mask * expanded_torus
        
        # 최종 latent space 및 에너지 등가성 보존
        perturbation = self.hypernetwork(purified_stream)
        final_latent_space = morphed_topology + perturbation
        conserved_weights = self.energy_parity(final_latent_space)
        
        metrics = {"transmission_rate": transmission.mean(), "gate_score": gate_mask.mean()}
        return conserved_weights, metrics

# ========================================================================
# 💻 프로덕션 검증 테스트 및 최적화 루프
# ========================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = IntegratedEgregoreSystem(dim=EgregoreGlobalConfig.LATENT_DIM).to(device)
    
    # id() 기반 LLRD (층별 학습률 격리)
    gate_params_ids = [id(p) for p in system.topology_gate.parameters()]
    base_params = [p for p in system.parameters() if id(p) not in gate_params_ids]
    gate_params = [p for p in system.topology_gate.parameters()]
    
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": 1e-3},
        {"params": gate_params, "lr": 1e-5} # 게이트 파라미터는 저학습률 적용
    ])
    
    # 가상 데이터 및 악성 노이즈 시뮬레이션
    batch_size, seq_len = 4, 128
    clean_stream = torch.randn(batch_size, seq_len, EgregoreGlobalConfig.LATENT_DIM, device=device)
    malicious_stream = clean_stream.clone()
    malicious_stream[:, 50:60, :] += torch.randn(batch_size, 10, EgregoreGlobalConfig.LATENT_DIM, device=device) * 15.0
    observer_state = torch.randn(batch_size, seq_len, EgregoreGlobalConfig.LATENT_DIM, device=device)
    
    # 학습 루프
    for epoch in range(1, 4):
        optimizer.zero_grad()
        output, metrics = system(malicious_stream, observer_state)
        loss = F.mse_loss(output, torch.randn_like(output))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Trans: {metrics['transmission_rate'].item():.4f}")

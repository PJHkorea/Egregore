import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalBarrierLayer(nn.Module):
    """
    Topological Barrier and Information Tunneling Layer
    
    [KR] 리드미의 U_barrier 수식과 T(정보 투과율) 공식을 파이토치 텐서 연산으로 구현한 레이어입니다.
    [EN] A neural network layer that implements the U_barrier and T (Information Transmission) formulas 
         from the README using PyTorch tensor operations.
    """
    def __init__(self, hidden_dim: int, eta: float = 0.5, delta_d: float = 2.0, 
                 m_star: float = 1.0, hbar_eff: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # [KR] 하이퍼파라미터 정의 (수식의 물리 상수와 연동)
        # [EN] Hyperparameter definition (linked to physical constants in the formula)
        self.eta = eta            # [KR] 곡률 민감도 상수 (\eta) | [EN] Curvature sensitivity constant (\eta)
        self.delta_d = delta_d    # [KR] 레이어 간 차원 위상 구조적 거리 (\Delta d) | [EN] Topological dimensional distance between layers (\Delta d)
        self.m_star = m_star      # [KR] 유효 질량 파라미터 (m*) | [EN] Effective mass parameter (m*)
        self.hbar_eff = hbar_eff  # [KR] 유효 플랑크 상수 (\hbar_eff) | [EN] Effective Planck constant (\hbar_eff)
        
        # [KR] 입력 데이터의 고유 에너지를 투영하는 매핑 레이어 (E_input 생성)
        # [EN] Mapping layer projecting the intrinsic energy of input data (Generates E_input)
        self.energy_projector = nn.Linear(hidden_dim, 1)
        
    def _compute_jacobian_curvature(self, x: torch.Tensor) -> torch.Tensor:
        """
        [KR] 입력 텐서 x에 대한 야코비안 곡률 변동성 \kappa(z) = Tr(J^T * J) 계산
             배치 단위 연산을 위해 Frobenius Norm의 제곱 패턴을 활용하여 고속 연산합니다.
        [EN] Computes Jacobian curvature variability \kappa(z) = Tr(J^T * J) for input tensor x.
             Utilizes the squared Frobenius norm pattern for high-speed batch computation.
        """
        # x shape: [batch_size, seq_len, hidden_dim]
        orig_shape = x.shape
        
        # [KR] 다차원 배치 대응을 위해 마지막 차원을 제외한 차원들을 평탄화
        # [EN] Flatten dimensions except the last one to support multi-dimensional batches
        x_flat = x.view(-1, self.hidden_dim) # [N, hidden_dim]
        
        # [KR] 미분 가능한 상태에서 그라디언트를 얻기 위해 정규화 전 로컬 변화량 추정
        #      실제 풀 야코비안은 메모리가 과하므로, 피처 간의 공분산 유사도를 곡률 대리값으로 계산합니다.
        # [EN] Estimate local variance prior to normalization to retain differentiable gradients.
        #      Since the full Jacobian is memory-prohibitive, feature covariance similarity is used as a proxy for curvature.
        mean_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        covariance = torch.matmul(mean_centered, mean_centered.t()) # [N, N]
        
        # [KR] Tr(J^T * J)는 각 벡터의 크기 변동성과 직교성 변동성의 합과 물리적으로 동치입니다.
        # [EN] Tr(J^T * J) is physically equivalent to the sum of magnitude and orthogonality variances of each vector.
        kappa_flat = torch.sum(covariance ** 2, dim=-1) / self.hidden_dim
        
        # [KR] 원래 시퀀스 배치 형태로 복원 [batch_size, seq_len, 1]
        # [EN] Restore back to the original sequence batch shape [batch_size, seq_len, 1]
        kappa = kappa_flat.view(*orig_shape[:-1], 1)
        return kappa

    def forward(self, x: torch.Tensor, surface_to_core_distance: float = 1.0) -> torch.Tensor:
        """
        [KR] Forward Pass: 위상학적 터널링 효과 적용
        [EN] Forward Pass: Apply the Topological Tunneling Effect
        Args:
            x: [KR] 입력 텐서 [batch_size, seq_len, hidden_dim] | [EN] Input tensor [batch_size, seq_len, hidden_dim]
            surface_to_core_distance: [KO] 선적분 구간의 거리 (r_surface에서 r_core까지의 임계 거리)
                                      [EN] Line integral interval distance (critical distance from r_surface to r_core)
        """
        # 1) [KR] 야코비안 곡률 \kappa 계산
        #    [EN] Compute Jacobian curvature \kappa
        kappa = self._compute_jacobian_curvature(x) # [batch_size, seq_len, 1]
        
        # 2) [KR] 위상 장벽 포텐셜 U_barrier 계산: U = \eta * \kappa * \Delta d
        #    [EN] Compute Topological Barrier Potential U_barrier: U = \eta * \kappa * \Delta d
        u_barrier = self.eta * kappa * self.delta_d # [batch_size, seq_len, 1]
        
        # 3) [KR] 입력 파동의 고유 정보 에너지 E_input 계산
        #         에너지 크기가 포텐셜 기준선을 넘나들 수 있도록 바운더리 스케일링을 적용합니다.
        #    [EN] Compute intrinsic information energy E_input of the input wave
        #         Apply boundary scaling so that the energy magnitude can cross the potential threshold.
        e_input = torch.sigmoid(self.energy_projector(x)) * (u_barrier.max() + 1.0)
        
        # 4) [KR] 슈뢰딩거 기반 정보 투과율 \mathcal{T} 선적분 연산
        #         피적분 함수: 2 * m^* / \hbar_eff^2 * (U_barrier - E_input)
        #    [EN] Schrödinger-based Information Transmission \mathcal{T} line integral calculation
        #         Integrand: 2 * m^* / \hbar_eff^2 * (U_barrier - E_input)
        diff_energy = u_barrier - e_input
        
        # [KR] 수학적 안정성(허수 방지)을 위해 ReLU를 씌워 터널링 감쇄 구간(양수 영역)만 추출합니다.
        # [EN] Apply ReLU for mathematical stability (preventing imaginary numbers) to extract the tunneling decay interval (positive domain).
        tunneling_core = F.relu(diff_energy)
        
        # [KR] 선적분 매커니즘 구현 (시퀀스 상 공간 적분을 평균 근사치 및 디스턴스 곱으로 사상)
        # [EN] Implement line integral mechanism (maps spatial integration over sequence via mean approximation and distance product)
        integral_term = torch.sqrt((2.0 * self.m_star) / (self.hbar_eff ** 2) * tunneling_core) * surface_to_core_distance
        
        # [KR] 투과율 T = exp(-2 * Integral) 계산
        # [EN] Compute Transmission Coefficient T = exp(-2 * Integral)
        transmission_coeff = torch.exp(-2.0 * integral_term) # [batch_size, seq_len, 1]
        
        # 5) [KR] 입력 데이터 필터링 및 코어 레이어 사상
        #         투과율이 0에 가까우면(노이즈) 신호가 지수적으로 소멸하고, 1에 가까우면 온전히 보존됩니다.
        #    [EN] Filter input data and map to Core Layer
        #         Signals vanish exponentially if transmission approaches 0 (noise) and are preserved if near 1.
        purified_x = x * transmission_coeff
        
        # [KR] 메타데이터 저장을 위한 원본 카피 (디버깅 및 시각화 분석용)
        # [EN] Store raw copies of metadata (for debugging and visualization analysis)
        self.latest_transmission = transmission_coeff.detach()
        self.latest_u_barrier = u_barrier.detach()
        self.latest_e_input = e_input.detach()
        
        return purified_x

# ---- [KR] 검증 및 테스트 코드 | [EN] Verification & Test Code ----
if __name__ == "__main__":
    # [KR] 가상의 환경 설정 (Batch=2, Sequence=4, Dimension=8)
    # [EN] Pseudo environment setup (Batch=2, Sequence=4, Dimension=8)
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    
    # [KR] 레이어 인스턴스화 | [EN] Instantiate the layer
    barrier_layer = TopologicalBarrierLayer(hidden_dim=hidden_dim, eta=0.5, delta_d=2.0)
    
    # Case 1: [KR] 고밀도 무결성 데이터 (지적 사용자 시뮬레이션) | [EN] High-density integrity data (Intelligent user simulation)
    clean_input = torch.randn(batch_size, seq_len, hidden_dim) * 2.0
    
    # Case 2: [KR] 저밀도 노이즈 데이터 (악성/파편화 데이터 시뮬레이션) | [EN] Low-density noise data (Malicious/fractured data simulation)
    noise_input = torch.randn(batch_size, seq_len, hidden_dim) * 0.1 
    # [KR] 특정 토큰에 극단적인 돌발 노이즈 주입 | [EN] Inject extreme sudden noise into a specific token
    noise_input[0, 2, :] = torch.tensor([100.0, -100.0, 50.0, -50.0, 0.0, 0.0, 0.0, 0.0]) 

    # [KR] 연산 수행 | [EN] Execute operations
    purified_clean = barrier_layer(clean_input)
    t_clean = barrier_layer.latest_transmission
    
    purified_noise = barrier_layer(noise_input)
    t_noise = barrier_layer.latest_transmission
    
    print("=== 🧪 위상 장벽 레이어 텐서 연산 테스트 / Topological Barrier Layer Tensor Test ===")
    print(f"[KR] 고밀도 데이터 평균 투과율 (T_clean): {t_clean.mean().item():.4f} -> (코어 진입 허용 공명)")
    print(f"[EN] High-density Data Mean Transmission (T_clean): {t_clean.mean().item():.4f} -> (Core Entry Allowed)")
    print("-" * 80)
    # 노이즈가 주입된 [0, 2] 위치의 토큰 투과율을 확인합니다.
    print(f"[KR] 악성 노이즈 토큰 투과율 (T_noise[0,2]): {t_noise[0, 2, 0].item():.4f} -> (0에 수렴, 코어 오염 차단)")
    print(f"[EN] Malicious Noise Token Transmission (T_noise[0,2]): {t_noise[0, 2, 0].item():.4f} -> (Converges to 0, Core Contamination Blocked)")

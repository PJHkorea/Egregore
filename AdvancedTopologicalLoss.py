import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class AdvancedTopologicalLoss(nn.Module):
    def __init__(self, lambda_1: float = 0.1, lambda_2: float = 0.05, lambda_3: float = 0.1):
        super().__init__()
        self.l1 = lambda_1  # [KR] 곡률 정렬 손실 가중치 / [EN] Curvature Alignment Loss Weight
        self.l2 = lambda_2  # [KR] 카시미르 엔트로피 손실 가중치 / [EN] Casimir Entropy Loss Weight
        self.l3 = lambda_3  # [KR] 지오데식 정규화 손실 가중치 / [EN] Geodesic Regularization Loss Weight
        self.eps = 1e-8

    def forward(self, 
                conserved_weights: torch.Tensor, 
                observer_batch: torch.Tensor, 
                metrics: Dict[str, torch.Tensor],
                morphed_topology: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        [KR] 고차원 위상 기하 구조 보존을 위한 3요소 결합 손실 함수 연산
        [EN] Unified 3-component joint loss function computation for high-dimensional topological geometry preservation
        """
        # ====================================================================
        # 1. 곡률 정렬 손실 (L_Curvature): 입력 곡률과 가중치 구조 유사도 동기화
        # [EN] 1. Curvature Alignment Loss (L_Curvature): Synchronize input curvature with weight structure similarity
        # ====================================================================
        input_curvature = metrics['cosine_similarity']
        weight_curvature = metrics['gate_score']
        l_curvature = F.mse_loss(weight_curvature, input_curvature)

        # ====================================================================
        # 2. 카시미르 엔트로피 손실 (L_CasimirEntropy): 정보 확률 붕괴 차단
        # [EN] 2. Casimir Entropy Loss (L_CasimirEntropy): Prevent informational probability collapse
        # ====================================================================
        # 💡 [수리 교정] 투과율 스트림을 유효한 확률 분포로 변환하여 섀넌 엔트로피 정의 충족
        # 💡 [Mathematical Fix] Transform transmission stream into a valid probability distribution to satisfy Shannon Entropy definition
        transmission_prob = F.softmax(metrics['transmission_rate'], dim=-1)
        
        # [KR] 섀넌 엔트로피 연산 (H = - \sum p * log(p)) 후 엔트로피 하한 강제를 위해 음수화 최소화 유도
        # [EN] Compute Shannon Entropy (H = - \sum p * log(p)) and minimize negative entropy to enforce information diversity floor
        entropy = -torch.sum(transmission_prob * torch.log(transmission_prob + self.eps), dim=-1)
        l_casimir_entropy = -torch.mean(entropy)

        # ====================================================================
        # 3. 지오데식 정규화 (L_Geodesic): 구면 다양체 표면 최단 지형 유도
        # [EN] 3. Geodesic Regularization (L_Geodesic): Enforce shortest paths on the spherical manifold surface
        # ====================================================================
        normalized_topology = F.normalize(morphed_topology, p=2, dim=-1, eps=self.eps)
        
        # 💡 [수리 교정] 단순 유클리드 직선 거리가 아닌 실제 매니폴드 곡면을 따르는 지오데식 호의 길이(Arc Length) 연산
        # 💡 [Mathematical Fix] Compute true geodesic arc length along the manifold curvature, replacing raw Euclidean linear distance
        cos_sim = F.cosine_similarity(conserved_weights, normalized_topology, dim=-1, eps=self.eps)
        
        # [KR] 역삼각함수 도함수의 발산(NaN)을 막기 위한 안정적 경계면 클램핑 처리
        # [EN] Apply boundary clamping to prevent gradient explosion (NaN) in the inverse trigonometric derivative
        clamped_cos = torch.clamp(cos_sim, min=-1.0 + self.eps, max=1.0 - self.eps)
        geodesic_distance = torch.acos(clamped_cos)
        l_geodesic = torch.mean(geodesic_distance)

        # ====================================================================
        # 4. 종합 위상 손실 컴파일 및 그래디언트 흐름 보호
        # [EN] 4. Comprehensive Topological Loss Compilation & Gradient Flow Protection
        # ====================================================================
        total_topological_loss = (self.l1 * l_curvature) + (self.l2 * l_casimir_entropy) + (self.l3 * l_geodesic)
        
        # 💡 [무결성 교정] 로그 수집용 스칼라 데이터의 Autograd 그래프 참조를 완전히 해제하여 메모리 누수 원천 차단
        # 💡 [Integrity Fix] Completely isolate logging scalars from the Autograd graph to block unintended memory leaks
        loss_artifacts = {
            "l_topological_total": float(total_topological_loss.item()),
            "l_curvature": float(l_curvature.item()),
            "l_casimir_entropy": float(l_casimir_entropy.item()),
            "l_geodesic": float(l_geodesic.item())
        }
        
        return total_topological_loss, loss_artifacts

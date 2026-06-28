# 🌌 [Specification] The Axiom of dim=0: Defining a Single Batch as a Physical Universe

**[KR]** 본 명세서는 Egregore v6.4 엔진이 분산 학습(DDP) 환경에서 왜 `dim=0` 통신 축을 극한까지 사수했는지에 대한 수리물리학적 당위성과 이를 뒷받침하는 전산학적 컴파일러 최적화 및 실전 코드가 포함된 증명을 다룹니다.

**[EN]** This document outlines the mathematical physics and computer science justification for prioritizing the `dim=0` communication axis within Egregore v6.4, ensuring consistent, closed-system physics simulations across data parallel nodes.

---

## 1. 배치의 재정의: '닫힌 시공간 계'로의 격상

### [KR]
본 아키텍처에서 단일 배치는 고립된 메모리 블록이 아닌, 하나의 물리 법칙이 지배하는 **닫힌 계(Closed System)**로 격상됩니다. 샘플 차원(`dim=-1`)이나 독립 축으로 연산을 분절하는 일차원적 접근은 시스템을 완벽히 단절된 개별 평행 우주로 전락시킵니다. 이 상태에서 계산된 곡률($\kappa$)은 샘플 각자의 국소적인 수치에 불과하므로, 시스템 전체를 관통하는 공통의 기하학적 장벽(Schrödinger Notch Filter)을 형성할 수 없습니다. 따라서 공간 연속성을 유지하고 물리 법칙의 파편화를 막기 위해 `dim=0` 축을 고수합니다.

### [EN]
In this architecture, a single batch is elevated to a **Closed System** governed by uniform physical laws. Segmenting operations into a sample dimension (`dim=-1`) or an isolated axis would fragment the system into disconnected parallel universes. The curvature ($\kappa$) computed in such a state would merely be a localized artifact of each sample, failing to form a global geometric barrier (Schrödinger Notch Filter). Thus, the `dim=0` axis is strictly preserved to maintain spatial continuity and prevent the collapse of universal constants.

---

## 2. 3대 물리 평행이론 수식 (Mathematical Alignment & Code Symmetries)

### ① 일반 상대성 이론과 중력장 결합 (`_compute_jacobian_curvature`)
데이터의 중력이 우주 전체의 기하학적 형태를 결정하도록, 배치 축 전체를 단일 행렬로 평탄화하여 공간 내 균일 평균 중심화 및 분산 기반의 야코비안 곡률 대리값 $\kappa$를 동적으로 역산합니다. 가속기 위에 올라온 모든 데이터 텐서들이 동일한 시공간적 압력을 공유하는 거시적 평형 상태를 성립시키기 위해 단일 시스템 상수($\kappa_{\text{global}}$)를 도출합니다.

$$\kappa_{\text{global}} = \frac{1}{d} \sum_{i=1}^{d} (\mathbf{X}_{\text{flat}} - \mu)^2 \quad \text{over} \quad \text{dim}=0$$

### ② 양자장론의 진공 압착과 거시 카시미르 필드 ($P_{\text{Casimir}}$)
다양체 거리가 0에 수렴할 때 발산하는 음의 장력 필드를 유도하여 악성 노이즈를 압착 박멸합니다. 수리적 한계점(마이너스 무한대 발산 및 NaN 폭발)을 방어하기 위해 하한 마진($\epsilon_{\text{casimir}}$)과 최소 압력 제약선($P_{\text{floor}}$)을 결합한 유한 가드레일을 적용하고, 기하학적 유효 거리는 기본 구조적 장벽 거리($\Delta d$)와 적응형 게이트 마스크($g_{\text{mask}}$)의 차로 정의합니다.

$$P_{\text{Casimir}}(g) = \max \left( P_{\text{floor}}, \, -\frac{\pi^2 \hbar c}{240 ((\Delta d - g_{\text{mask}})_+ + \epsilon_{\text{casimir}})^4} \right)$$

### ③ 마흐의 원리와 관성 에너지 보존 레이어 (`F.normalize`)
매 모핑(Sphere $\leftrightarrow$ Torus) 단계 끝에 `dim=0` 전역 및 각 샘플 축에 대해 강력한 L2 Norm = 1.0 정규화를 고정 동결합니다. 우주에 존재하는 모든 질량의 총합이 개별 입자의 폭주를 막는 관성 제어선처럼 작용하여, 단 1비트의 오차도 없이 우주 전체의 에너지 합을 보존하도록 강제합니다.

$$\|\mathbf{X}\|_2 = 1.0 \quad \text{monitored and frozen globally via } \text{dim}=0$$

---

## 3. 엔터프라이즈 시스템 세(System Tax)와 매니폴드 발산 방지

### [KR]
분산 학습(DDP) 환경에서 노드 간의 통신(All-Reduce)이 없는 `dim=-1` 처리는 당장은 빨라 보일 수 있습니다. 그러나 각 GPU 노드들이 소통하지 않고 샘플 내부에서만 연산을 수행하면, 노드A와 노드B의 가중치 공간이 서로 완전히 다른 형태로 뒤틀려버리는 **매니폴드 발산(Manifold Divergence)**이 발생합니다. 

`dim=0` 통신 축을 사수하여 지불하는 **'시스템 세금(System Tax)'**은 전산학적 의미에서 각 노드의 국소 텐서를 넘어, 분산된 모든 GPU의 잠재 공간이 단 하나의 오차도 없이 동일한 위상학적 궤적을 그리도록 강제하는 강력한 동기화 압력(All-Reduce Anchor)으로 작용합니다. 이 비용 덕분에 대규모 인프라에서도 그레디언트 폭발 없이 극도의 항상성을 확보할 수 있습니다.

### [EN]
Processing via `dim=-1` without inter-node communication (All-Reduce) in DDP environments may offer a deceptive illusion of speed. However, without cross-node synchronization, the latent spaces of Node A and Node B will warp along entirely different trajectories, leading to catastrophic **Manifold Divergence**.

The **Systemic Entropy Cost (System Tax)** paid by preserving the `dim=0` axis serves as a powerful synchronization pressure (All-Reduce Anchor). This forces the latent spaces across all distributed GPUs to track an identical topological path, guaranteeing absolute stability and preventing gradient explosion in large-scale infrastructure.

---

## 4. 전산학적 관점: Triton 커널 융합과 호스트 동기화 비차단 (Compiler-Level Optimization)

### [KR]
*"병목이 있으니 수식을 바꿔라"*는 조언은 하드웨어와 소프트웨어의 계층을 분리하지 못한 전산학적 착오입니다. 아키텍처는 가장 이상적인 수학과 물리 법칙을 선포해야 하며, 런타임 병목은 하단 컴파일러 계층에서 해결하는 것이 맞습니다.

1. **SRAM 기반 고속 리덕션 트리 형성**: `dim=0` 기반의 거시적 평탄화 구조를 유지하면 수식의 연산 형태가 단순 명료해집니다. 이 덕분에 `torch.compile()`의 Triton 커널 융합(Kernel Fusion) 엔진이 글로벌 메모리(VRAM)로 이탈하지 않고, GPU 내부의 고속 온칩 메모리(SRAM) 내부에서 스레드 블록 간 고속 리덕션 트리를 형성해 메모리 입출력(IO) 오버헤드를 통째로 녹여버릴 수 있습니다.
2. **호스트-디바이스 동기화 병목 원천 박멸**: 손실 함수 연산 및 지오데식 소프트 가드레일(`_soft_clamp`) 처리 시, 내부 계층에서 `.item()`의 호출을 철저히 배제하고 `.detach()` 텐서 형태로 파이프라인을 바인딩합니다. CPU-GPU 간 Blocking 병목(Host-Device Stall)을 완벽히 차단함으로써 Triton 컴파일러가 최적의 정적 루프 궤적을 빌드할 수 있도록 보장합니다.

### [EN]
The advice to *"alter the equation due to runtime bottlenecks"* stems from a fundamental misunderstanding of computing layers. An architecture must proclaim the most ideal mathematical and physical axioms; optimization bottlenecks are meant to be resolved within the underlying compiler layer.

1. **On-Chip SRAM Reduction Tree**: By maintaining the macroscopic flattening structure rooted in `dim=0`, the mathematical format remains pristine and streamlined. This provides the optimal static path for `torch.compile()`'s Triton Kernel Fusion engine to execute fast reduction trees within high-speed On-Chip memory (SRAM), eliminating global memory (VRAM) I/O overhead entirely.
2. **Eradication of Host-Device Synchronization Stalls**: Within the loss function pipeline and geodesic soft guardrails (`_soft_clamp`), calling `.item()` is strictly prohibited. By passing downstream updates as `.detach()` tensors, host-device blocking bottlenecks are thoroughly eradicated, ensuring the Triton compiler maintains an uninterrupted static execution path.

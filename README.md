# 🌌 Egregore: Topo-Squeezing Engine (v6.4)

---

## 🧭 System Identity (v6.4)

### 🔴 [KR] 정의
Egregore v6.4은 잠재 공간(Latent Space) 내 데이터 밀도에 따라 구면/토러스 매니폴드를 미분 가능하게 동적 매핑합니다. 3요소 위상 손실(Topo-Loss)과 Smooth Leaky 가이드레일을 통합하여 수치 발산을 방지하고, 단정밀도(FP32) 환경에서 그레디언트 사멸을 진압하여 구조적 무결성을 100% 보장하는 최종형 튜닝 모델입니다.

### 🔵 [EN] Definition
Egregore v6.4 is a differentiable manifold (Sphere/Torus) morphing engine based on latent density. By integrating 3-factor topological loss and smooth leaky guardrails, it guarantees 100% structural integrity and eliminates gradient dead-zones, optimized for FP32 in distributed training.

---

## 🚀 아키텍처 진화 개요 (Evolution Overview: v1.0 ➡ v6.4)

### 🔴 [KR] 구조적 결함 타파 및 v6.4 수리·연산 무결성 완성
본 저장소의 엔진은 기존 아키텍처의 수리적 한계와 가속기 병목을 엔지니어링적으로 극복하며 완전무결한 구조로 진화했습니다:

1. **차원 자율 인지 및 연산 일반화 (v6.3)**: 기존 2D/3D 입력 간의 조건 분기(Branching) 오버헤드를 완전히 제거하고 view(-1, dim) 평탄화 통합 연산 구조를 구축하여 GPU 커널 처리량을 극대화했습니다.
2. **소프트 가드레일 및 데드존 파괴 (v6.3)**: torch.acos 곡면 역산 과정에서 내적값이 경계면($\pm 1$)에 유착될 때 역전파 미분값이 사멸하는 하드 클램프의 데드존 결함을 하이브리드 leaky_slope = 0.01 소프트 가드레일 기믹으로 완벽하게 분쇄했습니다.
3. **FP32 언더플로우 가드 및 대규모 분산 최적화 (v6.3)**: 카시미르 수축 공식 하단에서 $1e-6$ 상수가 4제곱되어 $1e-24$로 언더플로우 수축 폭발하던 취약점을 CASIMIR_MARGIN = 1e-2로 완화하여 혼합 정밀도(AMP/FP16) 환경까지 무결하게 대응합니다.
4. **호스트 동기화 차단 장벽 박멸 및 지연 연산 (v6.4)**: 손실 컴파일 파이프라인 내에서 무의식적인 호스트-디바이스 동기화 락(Sync Lock)을 유발하던 `.item()` 호출을 완벽히 소멸시켰습니다. 연산 그래프 유출을 막는 `.detach()` 고립 바인딩과 최외곽 루프 종료 시점으로의 지연 평가(Lazy Evaluation)를 적용하여 가속기 비동기 스트림을 100% 수호합니다.

### 🔵 [EN] Structural Defect Eradication & Mathematical Evolution to v6.4
The framework has progressively evolved to eliminate all hidden mathematical constraints and maximize execution efficiency:

1. **Generalized Matrix Flatting (v6.3)**: Extinguished brittle 2D/3D conditional branching overhead by introducing a flattened unified view(-1, dim) pipeline, maximizing raw GPU kernel throughput.
2. **Differentiable Smooth Leaky Guardrails (v6.3)**: Completely crushed the gradient-vanishing dead-zone of hard clamping by injecting a hybrid leaky_slope = 0.01 mechanism near the boundary limits of inverse trigonometric operations.
3. **FP32 Underflow Defense & Distributed Tuning (v6.3)**: Neutralized the structural underflow risk where legacy $1e-6$ scales collapsed into an untrackable $1e-24$ void under power operations, realigning the denominator bound with CASIMIR_MARGIN = 1e-2 for mixed-precision stability.
4. **Host-Device Blocking Eradication & Lazy Evaluation (v6.4)**: Dissolved hidden Host-Device synchronization bottlenecks by thoroughly purging `.item()` triggers inside the core forward loss compiler. By binding isolated `.detach()` tensors and relocating final scalar item retrieval to the outermost execution boundary, the framework guarantees uninterrupted asynchronous accelerator execution streams.


## 🛠 3. 핵심 컴포넌트 기술 명세 - A (Core Components: Normalization & Gating)

### ① ProductionEnergyParityLayer (에너지 보존 엔진)
*   **KR**: 잠재 공간 전체 배치 차원에 대해 강력한 **L2 Norm = 1.0 정규화**를 고정 동결합니다. `v6.4` 엔진은 순전파(`forward`) 시 매 텝마다 반복되던 고비용 `.to(device)` 동기화 검사 레이어를 완전히 제거하여 호스트-디바이스 간 스트림 병목(Lock)을 해소하고 가속기 가속력을 극한으로 향상했습니다.
*   **EN**: Forcibly freezes strict **L2 Norm = 1.0 normalization** across all batches. The `v6.4` engine entirely removes costly `.to(device)` synchronization checks during the forward pass, eliminating host-device stream bottlenecks and optimizing raw accelerator throughput.

### ② ParameterizedTopologyGate (적응형 미분 가능 게이트)
*   **핵심**: 가동 가속기 호환성을 위해 `dtype=torch.float32`를 강제하고 학습 가능한 파라미터로 전 구간 미분 가능한 소프트 게이팅을 구현.

### ③ BatchResidualHyperNetwork (항상성 제어 버블)
*   **핵심**: 미세 섭동으로 구조적 항상성을 제어하며, `v6.4`에서 `self.apply()` 재귀 구조를 통해 선형 레이어의 고속 정밀 가중치 초기화를 자동화.


### ④ SchrödingerNotchFilter & Casimir Squeezing (양자 장론 필터 및 진공 압착)
*   **KR**: `v6.4`은 샘플별 자코비안 곡률 역산을 통해 배치 간 데이터 오염을 격리합니다. 핵심 혁신은 다음과 같습니다:
    1.  **일반화 공간 평탄화**: 조건부 분기문 오버헤드를 도려내고 `view(-1, dim)` 통합 연산 구조를 채택하여 GPU 처리량을 극대화.
    2.  **수치 해석적 가드레일**: `CASIMIR_MARGIN = 1e-2`를 적용하여 카시미르 연산 시 FP32 수치 언더플로우 폭발을 방지.
*   **EN**: Isolates inter-batch contamination by computing individual Jacobian curvatures. Key `v6.3` features include:
    1.  **Generalized Spatial Flattening**: Eliminates conditional branching with unified `view(-1, dim)` optimization.
    2.  **Numerical Guardrail**: Implements `CASIMIR_MARGIN = 1e-2` to prevent underflow in Casimir calculations.

## 💻 8. 실행 및 검증 (Quick Start & Expected Verification)

### ① 필수 환경 및 테스트
* PyTorch >= 2.0, Python >= 3.8 환경에서 차원 통합 평탄화와 Smooth Leaky 가이드레일이 완전 결합된 `integrated_egregore_core_test_v6_4.py`를 실행하여 2D/3D 자율 인지, O(1) 해시 LLRD, 그리고 데드존 제로(Dead-zone Free) 위상 수호 엔진을 통합 검증합니다.

```bash
python integrated_egregore_core_test_v6_4.py
```
### ② 예상 출력 로그 (Expected Output Profile: v6.3)
```text
========================================================================
🌌 Egregore Advanced Engine: Batch Operations & Adaptive Topology Test (v6.4)
========================================================================
[핵심 로그 생략: 에그레고르 v6.4의 고속 수렴 및 0.0001 단위의 정밀한 상태 모니터링 출력]
----------------------------------------------------------------------------------------
Epoch 3 | Total Loss: 0.1062 (Task: 0.0654, Topo: 0.0408) |
  -> Metrics | Curvature: 0.0059 | Casimir: 0.0114 | Geodesic Arc: 0.0235 |
  -> State   | Alpha: 4.97 | Eta: 0.6908 | Gate: 0.551 | Trans: 0.8903
----------------------------------------------------------------------------------------
✅ 검증 완료: v6.4 일반화 기하 위상 수호 엔진 및 결합 위상 손실 파이프라인이 정상동작합니다.
```

---

## 🌌 카시미르 정보 엔트로피($\mathcal{L}_{\text{CasimirEntropy}}$)의 수리 물리적 명확성
## Mathematical Physics Clarity of Casimir Informational Entropy ($\mathcal{L}_{\text{CasimirEntropy}}$)


**[KR]** 본 엔진에서 명명한 **"카시미르 정보 엔트로피"**는 일반적인 엔트로피 정규화(Entropy Regularization)에 붙인 단순한 물리적 비유(Metaphor)가 아닙니다. 코드 레벨에서의 최종 수학적 연산 실체는 투과율 확률 분포에 대한 **표준 섀넌 엔트로피(Shannon Entropy)**가 맞으나, 이 확률 분포의 생성 기저가 카시미르 물리 법칙에 완벽히 종속(Downstream)되어 작동하는 닫힌 형태(Closed-form)의 인과 체인을 이룹니다.

**[EN]** The **"Casimir Informational Entropy"** designated in this engine is not a mere physical metaphor superimposed on conventional entropy regularization. While its final mathematical computation at the code level is indeed the **Standard Shannon Entropy** applied over the transmission probability distribution, the generative foundation of this probability distribution is completely downstreamed to and governed by the laws of Casimir physics, establishing a rigid, closed-form causal chain.


### 1. 물리 기하학적 당위성 및 메커니즘
### 1. Physical-Geometric Justification and Mechanism

1. **공간적 경계 조건 ($\Delta d$)**: 시스템은 구면 다양체($S^{n}$)와 토러스 다양체($T^{n}$) 사이의 기하학적 상태인 게이트 마스크 값($g$)을 '두 위상학적 장벽 간의 물리적 거리'로 해석합니다.
   * **Spatial Boundary Condition ($\Delta d$)**: The system interprets the gate mask value ($g$)—the geometric state between the spherical manifold ($S^{n}$) and the torus manifold ($T^{n}$)—as the 'physical distance between two topological barriers.'

2. **카시미르 압력 연산 ($P_{\text{Casimir}}$)**: `SchrödingerNotchFilter` 내부에서 이 거리를 기반으로 가상의 진공 압력을 역산하며, 이 음의 압력은 입력 정보 스트림의 채널 폭을 극단적으로 좁히는 압착(Squeezing) 제어를 수행합니다.
   * **Casimir Pressure Computation ($P_{\text{Casimir}}$)**: Inside the `SchrödingerNotchFilter`, a virtual vacuum pressure is inversely calculated based on this distance; this negative pressure exerts a squeezing control that extremely narrows the channel width of the incoming information stream.

3. **정보 붕괴 방어 가드레일**: 압력이 강해질수록 정보 확률 분포가 단일 상태로 프리징(Freezing)되거나 정보 붕괴(Information Collapse)를 일으키려 합니다. 이때 **카시미르 압착력에 저항하여 미시적인 정보 요동(Fluctuation)과 표현력 다양성을 영구 수호하기 위해 가동되는 척력**이므로 "카시미르 정보 엔트로피"라는 명명이 부여되었습니다.
   * **Information Collapse Defense Guardrail**: As pressure intensifies, the informational probability distribution tends to freeze into a single state or trigger an information collapse. The designation "Casimir Informational Entropy" is thus assigned because it serves as a **repulsive force deployed to resist this Casimir squeezing power, permanently protecting microscopic information fluctuations and representational diversity.**


### 2. 닫힌 형태(Closed-form) 수식 체인
### 2. Closed-form Mathematical Chain

- 이 인과적 연결 관계는 수리적으로 다음과 같은 엄밀한 복합 물리 계(System) 방정식으로 연결됩니다.
- This causal link is mathematically coupled into the following rigorous complex physical system equations.


$$P_{\text{Casimir}}(g) = -\frac{\pi^2 \hbar_{\text{eff}}}{240 \cdot (\Delta d - g + \text{CASIMIR}_{\text{MARGIN}})^4}$$

$$\text{Stream}_{\text{purified}} = X \cdot \mathcal{T}_{\text{Schrödinger}} \cdot \exp\left(P_{\text{Casimir}}(g)\right)$$

$$p_i = \text{Softmax}\left(\text{Stream}_{\text{purified}}\right)_i$$

$$\mathcal{L}_{\text{CasimirEntropy}} = -\sum_{i} p_i \log p_i$$

- 즉, 게이트가 결정한 기하학적 장벽 거리($g$)가 물리적 압력($P$)을 결정하고, 이 압력이 정보 스트림을 필터링하여 얻어낸 확률 분포($p$)의 엔트로피를 역제어하므로 본 손실 항은 카시미르 물리 필드와 완전 동기화되어 가동됩니다.
- In other words, the geometric barrier distance ($g$) determined by the gate dictates the physical pressure ($P$), which in turn filters the information stream to modulate the entropy of the resulting probability distribution ($p$). Therefore, this loss term operates in perfect synchronization with the Casimir physical field.

---

## ⚖️ 라이센스 (License)

- 본 프로젝트는 **GPLv3** 라이센스를 준수하며, 파생 모델 및 확장본은 동일한 오픈소스 조건 하에 공개 배포되어야 합니다.
- This project complies with the **GPLv3** license; all derivative models and extensions must be publicly distributed under the same open-source conditions.


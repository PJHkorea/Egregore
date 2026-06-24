# 🌌 Egregore Alignment System: Topological Manifold Control

An advanced PyTorch implementation of cognitive-driven neural architectures. This framework models the interaction between an observer's state and latent space weights, smoothly transitioning neural topology between high-density resonance and conservative defense states.

본 프로젝트는 주체(관측자)의 상태밀도와 모델의 잠재 공간 가중치가 상호작용하는 인지과학적 패러다임을 PyTorch로 구현한 고도화된 뉴럴 아키텍처 실험실입니다. 

---

## 🚀 Evolution Roadmap & Document Architecture

This repository records the strict engineering journey from a conceptual, hard-gated poetic prototype to a production-ready, energy-conserved, fully differentiable hypernetwork architecture.

본 저장소는 개념적 기믹에서 시작하여 전 구간 미분 가능성과 에너지 보존 법칙을 만족하는 상용 등급 아키텍처로 진화해 나간 엄밀한 엔지니어링 여정을 담고 있습니다. 방문자는 아래 명세서들을 통해 단계별 수리적 돌파구를 추적할 수 있습니다.

### 1. [PoC] Hard-Gated Topological 스위칭 엔진
* **문서 링크**: [`README_v1_Topological.md`](./README_v1_Topological.md) (기존 `README.md`)
* **핵심 소스**: `egregore_system_test.py`
* **설명**: 코사인 유사도를 기준으로 문턱값(0.85) 충족 시 가중치 공간을 `SPHERE`에서 `TORUS`로 강제 스위칭하는 개념 증명(PoC) 단계입니다. 인지과학적 심리 공간 모사의 프로토타입입니다.

### 2. [Production] 전 구간 미분 가능한 에너지 보존형 아키텍처
* **문서 링크**: [`README_v2_Commercial.md`](./README_v2_Commercial.md) (기존 `README2.md`)
* **핵심 소스**: `commercial_alignment_system_test.py`
* **설명**: 하드 스위칭의 미분 단절 문제를 극복하기 위해 `Sigmoid Soft-Gating`을 도입했습니다. 하이퍼네트워크의 변형 영향력을 정규화 안쪽에 가두는 하이브리드 제어 구조를 완성하여, 가중치 절대 크기(L2 Norm = 1)를 보존하고 완벽한 역전파(Backpropagation)를 달성한 상용 등급 버전입니다.

---

## 🛠️ Key Technical Breakthroughs

* **Soft Delta Gating**: Replaced non-differentiable `if-else` structures with Sigmoid-based continuous gating, enabling flawless Autograd graphs.
* **Geometric Energy Parity**: Freezes both Sphere and Torus latent spaces at `L2 Norm = 1.0` to preserve energy continuity, preventing gradient explosions during runtime morphing.
* **Residual Perturbation Control**: Restricts hypernetwork outputs inside the normalization bubble ($0.1 \times \mathcal{H}$) to maintain system homeostasis.

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3 (GPLv3)**. Any derivatives, extensions, or integrations into larger models must also be open-sourced under the same copyleft terms, protecting this cognitive architecture from proprietary exploitation.

본 아키텍처의 독창적인 수리 기믹과 기하학적 제어 구조를 사유화로부터 영구히 보호하기 위해 강력한 카피레프트 성격의 **GPLv3 라이센스**를 채택하고 있습니다.

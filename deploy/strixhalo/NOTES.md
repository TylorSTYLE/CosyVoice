# NOTES — CosyVoice on Strix Halo (gfx1151) : findings·결정·벤치 로그

> 실행 모델: DEV=Apple Silicon M2(코드/문법/mock 테스트만). TARGET=Strix Halo 서버(사용자가
> 서버에서 직접 실행, SSH 불가). 서버 출력을 붙여받아 여기 기록한다. 수치엔 출처를 남긴다.

---

## Phase 0 — 정찰 / 토폴로지 확정

### 로컬(Mac) 정찰 결과 (2026-07-09, 이 세션)
- 레포에 `deploy/` 없음 → `deploy/strixhalo/` 신규. fork=`TylorSTYLE/CosyVoice`(HTTPS).
  서브모듈 `third_party/Matcha-TTS`.
- vLLM 연동: `AutoModel(load_vllm=True)` → `cosyvoice/cli/model.py:281 load_vllm()` →
  `export_cosyvoice2_vllm()` 후 `LLMEngine.from_engine_args(EngineArgs(model=dir,
  skip_tokenizer_init=True, enable_prompt_embeds=True, gpu_memory_utilization=0.2))`.
  → **`enable_prompt_embeds=True` 가 CosyVoice 고유 요구** (Phase 2 vLLM 빌드 검증 포인트).
  vLLM 모델 클래스 `cosyvoice/vllm/cosyvoice2.py:CosyVoice2ForCausalLM`(Qwen2), V1/legacy 분기 존재.
- 버전(README `#### vLLM Usage`): vLLM `0.9.0`(legacy, transformers==4.51.3) 또는 `0.11.0`
  (V1, transformers==4.57.1), numpy==1.26.4. 0.10.x 미검증. base torch==2.3.1+CUDA(→ROCm 교체), py3.10.
- 파이프라인: LLM(vLLM) → flow(DiT, torch) → hift(HiFi-GAN, torch).
- 기존 서빙 `runtime/python/fastapi/server.py` = Form 기반 비-OpenAI → 신규 OpenAI `server.py` 필요.
- 모델 = **Fun-CosyVoice3-0.5B-2512**(우선), `vllm_example.py:22` 기준 load_vllm=True, fp16=False.
- asset/ 에 KR 클립 없음 → **우선 기존 프롬프트 클립으로 한국어 텍스트 검증**.
- 결정: 코드동기화=git, 서버접속=사용자 직접 실행(SSH 불가).

### 서버 정찰 결과 (2026-07-09, 서버 로컬 실행 — SSH 불가)
- **ARCH/OS/커널**: x86_64 · Ubuntu **26.04 LTS (Resolute Raccoon)** · 커널 **7.0.0-22-generic**
  → 최신 KFD ABI, gfx1151 지원 유리.
- **GPU**: AMD RYZEN AI Max+ 395 w/ Radeon 8060S, **gfx1151**(40 CU) 확정. rocm-smi GFX Version=gfx1151.
  (rocminfo에 `gfx11-generic` 타깃도 노출 — 폴백 아키텍처.)
- **디바이스**: `/dev/kfd`(root:**render** 235,0), `/dev/dri/renderD128`(root:**render** 226,128),
  `/dev/dri/card1`(root:**video** 226,1). 전부 `crw-rw----`.
- **그룹 GID**: **render=993(비표준!)**, **video=44**. user `tylorstyle`은 양쪽 다 소속.
  → **결정**: 컨테이너 root 실행이면 device-only로 충분. 비-root거나 rw 실패 시
  `group_add: ["993","44"]`(render, video) 추가. compose에 device-only로 시작 후 실측.
- **호스트 ROCm**: `/opt/rocm` **없음**. 그러나 rocminfo/rocm-smi 동작 → 호스트는 **KFD 커널
  드라이버만** 제공. 컨테이너가 TheRock ROCm 7.x userspace 전부 반입 = **의도한 아키텍처와 일치**.
  (호스트-컨테이너 ROCm 버전 독립. 필요한 건 호스트 KFD가 gfx1151 enumerate 가능 → 확인됨.)
- **Docker**: 29.4.1. `docker info` 권한 실패 → ⚠️ 빌드 전 docker 그룹/sudo 권한 확인 필요(블로커 아님).
- **git/net/disk**: git 2.53.0, `/` 1.9T 중 **1.1T 여유(44% 사용)**, HF `HTTP/2 200`(접근 가능).

**Phase 0 게이트 판정: PASS.** 타깃 HW/드라이버/네트워크/디스크 모두 진행 가능. Ollama 컨테이너는
현재 미기동이라 device 설정 카피 불가 → device-only 시작 후 group_add 폴백 전략 채택.
다음: 정확한 ROCm/torch/vLLM 버전·URL 실조회 → Phase 1 Dockerfile.

---

## Phase 1 — gfx1151 torch (서버)

### 확정 스택 (버전·URL 실조회, 2026-07-09 — 출처 하단)
- **Base**: `ubuntu:26.04`.
- **ROCm userspace**: ~~TheRock S3 tarball~~ → **AMD prerelease pip 인덱스로 통일**(아래 1차 빌드
  실패 교훈). 인덱스 `https://rocm.prereleases.amd.com/whl/gfx1151/` 가 **런타임+devel 툴체인을 모두
  호스팅**: `rocm-sdk-libraries-gfx1151`(런타임), `rocm-sdk-devel`(hipcc/헤더, Phase 2 vLLM 빌드용),
  `rocm-sdk-core`. torch 휠이 self-contained → **시스템 `/opt/rocm` 불필요, S3 tarball 불필요**.
- **PyTorch**: prerelease 인덱스 네이티브 휠. Phase 1 핀 `torch==2.9.1+rocm7.13.0rc2`(cp310, 존재 확인).
  torch 가 `rocm-sdk-libraries-gfx1151` 를 dep 으로 자동 반입. **stock pytorch.org ROCm 휠 금지**
  (gfx1151 SIGSEGV). repo.radeon.com gfx1151 미제공(gfx1100 전용) → 사용 금지.
- **Python**: **3.10**(uv standalone). 인덱스 cp310 휠 존재. CosyVoice 생태계가 3.10 기준.
- **numpy**: 1.26.4 (torch 뒤 PyPI 에서 설치).

### 1차 빌드 실패 & 전환 (2026-07-09, 서버)
- 증상: `RUN install_rocm_sdk.sh` 에서 `curl: (6) Could not resolve host:
  therock-nightly-tarball.s3.amazonaws.com`. 동일 빌드에서 apt/astral.sh/ubuntu 는 정상 resolve
  → 일반 DNS 정상, **그 S3 버킷 호스트명이 틀림**(리서치 서브에이전트의 미검증 URL).
- 근본 대응: S3 tarball 경로 **폐기**. prerelease pip 인덱스가 런타임·devel 을 다 제공함을 실조회로
  확인(위) → `install_rocm_sdk.sh` 삭제, Phase 1 은 `uv pip install --pre torch==...` 만으로 구성.
- 교훈: ROCm 조달을 pip 인덱스 하나로 단일화(Phase 2 devel 도 `rocm-sdk-devel` pip 로).

### 2차 빌드 실패 & 전환 (2026-07-09, 서버)
- 증상: `uv pip install --pre torch==2.9.1+rocm7.13.0rc2 --index-url <rocm> --extra-index-url pypi`
  → `No solution found ... torch was found on pypi but not at the requested version`. uv 기본
  first-index 전략이 `torch` 를 pypi 에서 먼저 잡아 `+rocm` local 버전을 못 찾음.
- 대응: **pypi extra 제거, rocm 인덱스 단독**. 그 인덱스가 torch deps + 런타임까지 전부 호스팅 확인됨.
  numpy==1.26.4 는 스모크 불필요 → Phase 1 에서 제거(CosyVoice 런타임 스테이지로 이동).
- **핵심 gfx1151 env**: `PYTORCH_ROCM_ARCH=gfx1151`, `HSA_OVERRIDE_GFX_VERSION=11.5.1`(하이브리드
  안전값, 네이티브면 무해), `VLLM_ROCM_USE_AITER=0`(CDNA 전용 커널 → gfx1151 프리즈 회피),
  `ROCBLAS_USE_HIPBLASLT=1`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `HIP_FORCE_DEV_KERNARG=1`.

### 만든 파일 (로컬)
- `Dockerfile`(멀티스테이지, 현재 `rocm-torch` 스테이지만 — SDK tarball 없이 pip 인덱스 torch) ·
  `smoke_gpu.py`(matmul 정합성 + AR 디코드 프록시 마이크로벤치) · 루트 `.dockerignore`.
  (`scripts/install_rocm_sdk.sh` 는 1차 빌드 실패 후 삭제 — pip 인덱스로 대체.)

### 서버 실행 결과 (2026-07-09) — 게이트 PASS ✅
빌드: `--target rocm-torch` 성공(torch+런타임 휠 다운로드 ~103s, 이미지 export ~208s, 총 313s).
스모크(`--device /dev/kfd --device /dev/dri`, device-only, group_add 불필요):
```
torch: 2.9.1+rocm7.13.0rc2 | hip: 7.13.99004-3309c6114a
device: Radeon 8060S Graphics
[matmul] max abs err vs CPU: 2.289e-04  (OK)
[AR proxy] per-step d2h sync :   12071.2 tok/s  (0.083 ms/tok)
[AR proxy] no per-step sync  :   13984.1 tok/s  (0.072 ms/tok)
[AR proxy] sync overhead x   : 1.16
SMOKE OK
```
- **세그폴트(exit 139) 없음** → 최대 위험 #1(stock 휠 SIGSEGV) 제거. device-only 패스스루로 GPU
  잡힘(sudo docker, group_add 없이). numpy 미설치 경고는 무해(Phase 1에서 의도적 제외).
- **⚠️ 중대 발견 — memcpy 병목 미재현**: 계획 대전제(AR 디코드의 92~95%가 hipMemcpyWithStream,
  PyTorch #171687)가 이 프록시에선 **sync overhead 1.16× (16%)** 로만 나타남. 해석:
  (A) 프록시가 단순(작은 GEMV+argmax+.item())해 실제 CosyVoice AR 디코드(HF generate/KV캐시)를
  대표 못함, 또는 (B) ROCm 7.13 네이티브 휠이 #171687 을 이미 상당 완화. **프록시로 결론 불가.**
- **결정 영향**: vLLM 소스빌드(가장 위험·고비용 단계)에 뛰어들기 전에, **CosyVoice eager 파이프라인의
  실제 한국어 RTF 를 먼저 실측**해 vLLM 필요성을 판단하는 것이 합리적(실제 병목 측정 후 최적화).
  → Phase 2 방향을 사용자와 재조정(아래 Phase 2 로그 참조).

## Phase 2 — (재조정) eager CosyVoice RTF 먼저 측정 → vLLM 필요성 판단
사용자 결정(2026-07-09): Phase 1 의 memcpy 병목 미재현(sync 1.16×)을 근거로, 가장 위험·고비용인
vLLM 소스빌드에 앞서 **eager CosyVoice3 로 실제 한국어 RTF 를 먼저 실측**. eager RTF 는 어차피
vLLM 비교의 분모. RTF<1.0 이면 vLLM 을 건너뛸 수도.

### 만든 파일 (로컬)
- Dockerfile `cosyvoice-eager` 스테이지(rocm-torch 위, vLLM/TRT/JIT 없음).
- `requirements-strix.txt`(requirements.txt 에서 CUDA/TRT/train 의존성 제거, onnxruntime-gpu→CPU).
- `constraints-strix.txt`(torch 2.9.1 / torchaudio 2.9.0 / numpy 1.26.4 고정 — 런타임 deps 설치가
  CUDA torch 로 바꾸지 못하게).
- `bench_cosyvoice.py`(전체 파이프라인, `inference_cross_lingual` 로 KR 합성, RTF 측정+wav 저장).

### 근거 (코드 실측)
- deepspeed=학습 전용(bin/train.py), tensorrt=load_trt 시 lazy import → eager 에서 불필요.
- frontend onnx: campplus=CPU 고정, speech_tokenizer=cuda면 CUDAExecutionProvider 시도→CPU onnxruntime
  라 경고 후 CPU 폴백(core 무수정). openai-whisper 는 frontend 최상단 import → 필수.
- CosyVoice3.__init__ 은 `load_jit` 인자 없음 → bench 는 load_trt/load_vllm/fp16 만 전달.
- torchaudio 2.9.0(인덱스) vs torch 2.9.1 patch 불일치 → `--no-deps` 설치로 회피.

### 서버 실행 결과 (붙여넣기 대기)
```
# build --target cosyvoice-eager  +  모델 다운로드  +  bench_cosyvoice.py RTF
```
게이트: 알아들을 수 있는 KR wav + best RTF. RTF<1.0 이면 vLLM 불필요 가능성(→ Phase 4 서빙 직행),
RTF≥1.0 이면 vLLM 소스빌드로 최적화 진행.

## Phase 2b — vLLM 소스빌드 (필요 판정 시, GO/NO-GO)
(eager RTF 결과에 따라)

## Phase 3 — CosyVoice 연동 (서버)
(대기)

## Phase 4 — 서빙 + 벤치 (서버/로컬)
(대기)

## Phase 5 — 마감
(대기)

---

## 출처 (URL·버전, 실조회 2026-07-09)
- hec-ovi/vllm-qwen (빌드 레퍼런스): https://github.com/hec-ovi/vllm-qwen
  - Dockerfile: https://raw.githubusercontent.com/hec-ovi/vllm-qwen/main/Dockerfile
  - install_rocm_sdk.sh: https://raw.githubusercontent.com/hec-ovi/vllm-qwen/main/scripts/install_rocm_sdk.sh
  - patch_strix.py: https://raw.githubusercontent.com/hec-ovi/vllm-qwen/main/scripts/patch_strix.py
- TheRock gfx1151 나이틀리 tarball(S3 리스팅):
  https://therock-nightly-tarball.s3.amazonaws.com?list-type=2&prefix=therock-dist-linux-gfx1151-7
- AMD gfx1151 prerelease torch 인덱스: https://rocm.prereleases.amd.com/whl/gfx1151/ (및 `/torch/`)
- TheRock: https://github.com/ROCm/TheRock (Discussion #655: https://github.com/ROCm/TheRock/discussions/655)
- repo.radeon.com gfx1151 torch: **미제공**(gfx1100/1101 전용). 근거 https://github.com/ROCm/ROCm/issues/5339
- vLLM V1 prompt-embeds(0.11.0 지원 근거): https://github.com/vllm-project/vllm/issues/22124 (PR #24278)
- Fun-CosyVoice3-0.5B-2512: HF https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 ·
  Modelscope https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512

## Phase 2 결정 대기 (go/no-go 이전)
- vLLM 버전: **0.11.0(V1) 우선**, 단 0.11.0 `requirements/rocm.txt` 가 torch==2.8.0 핀 → TheRock
  torch 2.9.1 과 충돌 가능. 실패 시 **HEAD(0.19.2rc1)** 로 폴백(gfx1151 실증된 유일 조합). 서버에서
  `git checkout v0.11.0` + `patch_strix.py` + `--constraint`(torch 2.9.1 핀) `--no-deps` 빌드로 먼저 시도.
- vLLM 빌드 툴체인: `pip install rocm-sdk-devel`(인덱스) 로 hipcc/clang/헤더 확보(S3 tarball 대신).
  ROCm 경로는 `rocm-sdk path --root` / `--bin` 으로 해석(설치 위치가 venv site-packages 하위).
- vLLM 빌드 env: `VLLM_TARGET_DEVICE=rocm`, `PYTORCH_ROCM_ARCH=gfx1151`, `HIP_ARCHITECTURES=gfx1151`,
  `GPU_TARGETS=gfx1151`, `CC/CXX=<rocm-sdk path --bin>/clang(++)`, `CMAKE_ARGS=-DGPU_TARGETS=gfx1151 ...`.
- 런타임: `--enforce-eager`(HIP graph 프리즈 회피), `--enable-prompt-embeds --skip-tokenizer-init`.
- 미검증 리스크: 0.11.0 vs torch 2.9.1 ABI, numpy 1.26.4 vs 최신 deps 충돌 → 서버에서 실측.

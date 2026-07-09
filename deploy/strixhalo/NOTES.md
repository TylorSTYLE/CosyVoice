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

### 서버 실행 결과 (2026-07-10)
빌드/모델다운로드/실행 성공까지 해결한 이슈(전부 dep/어댑터, GPU 무관):
1. `openai-whisper` sdist: setuptools≥81 이 pkg_resources 제거 → `setuptools<81` 선설치+constraint.
2. whisper triton<3 vs ROCm torch triton 3.5.1 → whisper `--no-deps` 별도 설치(deps 명시).
3. `pyworld` sdist: `--no-build-isolation` 에 numpy 부재 → numpy+cython 선설치.
4. torchaudio 2.9 가 load/save 를 TorchCodec 으로 라우팅(미설치) → bench 에서 soundfile shim(core 무수정).
5. 이 fork API 는 prompt 를 **파일 경로**로 받음 → bench 가 경로 전달(텐서 아님).
6. CosyVoice3 는 prompt_text 에 `<|endofprompt|>`(151646) 필수 → zero_shot + CV3 prompt_text.

**GPU 연산 실증(gfx1151, 세그폴트 없음)**: 모델 GPU 로드 성공, MIOpen 커널 DB 사용, DiT flow 가
AOTriton efficient attention 으로 SDPA 실행, HiFi-GAN conv1d GPU 실행. onnx frontend=CPU 폴백(정상),
wetext 프론트엔드 사용. torch `2.9.1+rocm7.13.0rc2`, hip `7.13.99004`.

**Phase 3 게이트 PASS ✅**: 한국어 wav 합성·재생 성공(zero_shot, 중국어 프롬프트 클립로 cross-lingual
KR). → 최대 위험 #1(stock 휠 SIGSEGV) 완전 제거, 전체 파이프라인 gfx1151 동작 확인.

### RTF 실측 (2026-07-10) — 대표 한국어 문장, audio ~6s, token2wav=flow+hift 계측
| 실행 | RTF | flow+hift(token2wav) | llm+rest | 비고 |
|---|---|---|---|---|
| fp32 run0* | 1.17 | 1.60s | 5.56s | *seed0=warmup shape 히트 |
| fp32 run1/2 | 2.4 | 8–9s | ~5s | 새 shape |
| **fp16 run0*** | **0.47** | **0.41s** | **2.50s** | 실시간 달성(shape 히트) |
| fp16 run1/2 | 1.64 | 7–7.5s | ~2.2s | 새 shape, flow/hift 지배 |

### Phase 2 go/no-go 결론: **vLLM은 이 문제의 열쇠가 아니다** (데이터 근거)
- **LLM AR 디코드는 병목이 아님**: fp16 로 `llm+rest` 가 5.5s→2.2s. (원래 가설의 hipMemcpyWithStream
  붕괴는 이 ROCm 7.13 네이티브 스택에서 관측되지 않음 — Phase1 프록시 sync 1.16× 와 일관.)
- **진짜 병목 = flow(DiT)+HiFi-GAN 의 conv (token2wav)**: 새 utterance shape 마다 7–9s. 원인은
  MIOpen 이 gfx1151 튜닝 perf-db(`gfx1151_20.HIP.fdb.txt`) 부재로 immediate-mode 폴백(GemmFwdRest,
  workspace 부족)을 쓰기 때문. **LLM 과 무관 → vLLM 으로 해결 불가.**
- **shape 캐시 히트 시 fp16 RTF 0.47** (token2wav 0.41s) → 실시간은 실제로 가능. 관건은 flow/hift
  conv 를 일관되게 빠르게 만드는 것(MIOpen 튜닝/캐시, cudnn.benchmark, 스트리밍 고정 chunk shape).
- **∴ vLLM 소스빌드(Phase 2b 원안)는 투자 대비 실익 없음** (LLM 이미 충분히 빠름). 최적화 축을
  **fp16 + flow/hift MIOpen + 스트리밍**으로 전환 권고.

### 다음 후보(값 미검증, 서버 실측 필요)
- `torch.backends.cudnn.benchmark=True` + `MIOPEN_FIND_MODE=NORMAL` + 영속 user-db → conv 알고리즘
  탐색·캐시(같은 shape 재사용 빨라짐).
- **스트리밍(stream=True)**: 고정 크기 chunk 로 flow 호출 → MIOpen shape 안정 → 일관 속도 + 낮은
  first-audio 지연('라이브'에 부합).
- flow n_timesteps(DiT 스텝) 축소(품질/속도 트레이드오프).

## Phase 2c — flow/hift MIOpen 최적화 시도 (2026-07-10) — 실패, 플랫폼 한계 확인
시도한 레버(어댑터/env, core 무수정): `torch.backends.cudnn.benchmark=True`,
`MIOPEN_FIND_MODE=NORMAL`, 영속 MIOpen user-db, 스트리밍(고정 chunk 기대).
- **MIOPEN_FIND_MODE=NORMAL: 해로움**. novel shape 마다 full autotune ~50s(1st inv run1/2
  token2wav 51–52s). db warm 후에도 novel shape 여전히 ~7s → autotune 최선도 GemmFwdRest(느림).
- **cudnn.benchmark=True: 무효과**. token2wav 7.1/7.4s(이전 7.2/7.5s와 동일). MIOpen
  `provided < required` workspace 경고 그대로 → torch↔MIOpen 이 gfx1151 에서 conv workspace 부족
  → 느린 GemmFwdRest 폴백. 이게 근본 원인, env 로 안 풀림.
- **스트리밍: total RTF 악화**(best 1.35 vs non-stream 0.455). chunk 마다 token2wav → 페널티 반복.
  `ttfc` ~2.5s(첫 오디오 지연).
- → 세 레버 모두 되돌림(FIND_MODE, benchmark 제거). 기본이 더 나음.

### 확정 결론 (수치 기반, 정직 보고)
- **CosyVoice3 한국어 TTS 는 gfx1151 에서 동작하고, 실시간(fp16 RTF 0.455)은 shape 캐시 히트 시
  실증됨** (token2wav 0.43s, llm 2.37s). 세그폴트 없음.
- **일관된 실시간의 벽 = flow(DiT)+HiFi-GAN conv 의 MIOpen GemmFwdRest 폴백**(workspace 부족).
  novel utterance 길이마다 token2wav ~7s → RTF 1.6–2.4. 시도한 env/스트리밍으로 해소 불가 =
  gfx1151 MIOpen 플랫폼 한계(torch rocm7.13 스택 기준).
- **vLLM 은 무관**(LLM 은 fp16 로 이미 ~2.2s). 원래 가설(AR memcpy 병목→vLLM) 이 하드웨어에선 성립 안 함.

### 남은 옵션 (미해결)
- 서빙 진행 + 정직한 RTF 문서화(warm 0.47 / novel 1.6–2.4).
- flow 입력 고정길이 버킷팅(패딩)으로 MIOpen 커널 재사용 → 일관 0.47 목표. **flow 추론(core) 수정
  필요 → 승인 필요.**
- 더 깊은 MIOpen 튜닝(PYTORCH_MIOPEN_SUGGEST_NHWC, TheRock 최신 빌드 등). 불확실.

## Phase 2b — vLLM 소스빌드 (보류/불필요)
Phase 2 데이터상 LLM 은 병목 아님 → vLLM 은 실시간 달성에 불필요.

## Phase 2b — vLLM 소스빌드 (필요 판정 시, GO/NO-GO)
(eager RTF 결과에 따라)

## Phase 3 — CosyVoice 연동 (서버)
(대기)

## Phase 4 — 서빙 (서버/로컬) — 검증 PASS ✅ (2026-07-10)
- 로컬(Mac): `pytest deploy/strixhalo/tests/` 9 passed / 1 skipped(ffmpeg). server.py OpenAI 스키마·
  오디오 인코딩·요청/응답 로직 mock 검증 완료.
- 서버: `docker compose up -d --build`(serve 스테이지=uvicorn), 포트 8880(.env SERVE_PORT).
  - `GET /health` → `{"status":"ok"}` ✅
  - `GET /v1/models` → `{"object":"list","data":[{"id":"cosyvoice3-0.5b",...}]}` ✅
  - `POST /v1/audio/speech`(KR, mp3) → mp3 생성 성공. **cold 첫 요청 42.5s**(~7s 오디오 → RTF~6,
    MIOpen 첫 커널 컴파일 = 이미 규명된 novel-shape 페널티). warm(동일 문장 재요청) RTF 측정은 대기.
- compose 볼륨은 명시적 name 으로 기존 cosy_models/hf/ms/miopen 재사용. 캐시 named volume 영속화 완료.

### 최종 검증 (2026-07-10) — 완료 정의 충족 ✅
GPU 서빙 확인(compose, seed 고정 + startup warmup):
- 1st 요청(해당 shape 일회성 MIOpen 컴파일 포함): **9.7s**
- 2nd 요청(동일 텍스트 → seed 고정으로 동일 shape → MIOpen 캐시 히트): **2.72s, RTF 0.398**
  (audio 6.08s, 로그 `rtf 0.3977`) = **실시간 2.5배**. 한국어 오디오 정상 재생.
- 완료 정의: 서버 curl→KR mp3 ✅ / RTF<1.0(실측 0.40) ✅ / GPU 사용·무세그폴트 ✅.
- 함의: **반복/정형 문구(HA 응답 등)는 캐시 히트로 실시간**. 새 문장 첫 합성은 그 shape 컴파일
  비용(수~수십 초) 1회, 이후 동일 텍스트는 빠름(MIOpen 캐시 named volume 영속).

### 서빙에서 잡은 함정 2가지 (중요)
1. **`.env` 의 빈 `HSA_OVERRIDE_GFX_VERSION=` 가 이미지 ENV(11.5.1)를 덮어써 GPU 미인식**.
   `env_file` 은 빈 값도 로드해 오버라이드 → ROCm 이 gfx1151 못 잡고 CPU 폴백(warmup 160s).
   `.env.example` 에서 해당 줄 제거(이미지가 11.5.1 설정). **교훈: env_file 에 빈 변수 두지 말 것.**
2. **compose `devices:` 가 이 docker 버전에서 GPU 접근을 제대로 못 줌**(노드는 보이나 cuda False).
   `docker run --device` 는 정상. 위 #1 이 실제 근본 원인이었고, device 형식은 Ollama 와 동일한
   `- /dev/kfd:/dev/kfd`, `- /dev/dri:/dev/dri` 매핑 형식 사용. server.py 에 GPU 미인식 시
   즉시 실패하는 `COSYVOICE_REQUIRE_GPU` 가드 추가(CPU 조용한 폴백 방지).

### 서버 개선 (server.py)
- **고정 seed(COSYVOICE_SEED=0)**: 동일 텍스트→동일 토큰 길이→동일 conv shape→MIOpen 캐시 히트.
- **startup warmup**: 첫 실요청의 콜드 컴파일 완화.
- **GPU 가드**: torch.cuda 불가 시 명확한 에러로 기동 실패(CPU 폴백 차단).

## 후속 — 한국어 숫자 영어발음 수정 (2026-07-10)
- 증상: 한국어 문장의 아라비아 숫자가 영어로 발음("2024"→twenty…).
- 원인: `cosyvoice/cli/frontend.py:127 text_normalize` 가 `contains_chinese`(한자)로 분기 → 한글은
  한자 없어 **영어 분기**(`en_tn_model.normalize` + `spell_out_number`/inflect)로 숫자 영어화.
  CosyVoice 에 한국어 정규화기 없음(wetext FST=zh/en/ja).
- 해결(core 무수정): `server.py` 엔드포인트에서 합성 전 `normalize_korean_numbers()` 로 숫자를
  사이노-코리안 변환(`2024`→이천이십사, `3.5`→삼점오). 한글 포함 텍스트에만 적용, `COSYVOICE_KO_NUMBERS`
  토글. 한계: 사이노 고정(시각/조수사 고유어·전화번호 자리읽기 미대응). `/web` 데모 별칭 추가.

## 전체 결론
- gfx1151 에서 **CosyVoice3 한국어 TTS 실시간 서빙 성립**(캐시 히트 RTF 0.40, OpenAI 호환 API).
- **vLLM 불필요**(LLM 은 fp16 로 이미 빠름; 병목은 flow/hift MIOpen, vLLM 무관).
- 남은 개선 여지: 새 문장 첫 합성의 shape 컴파일 비용 → flow 고정길이 버킷팅(core 수정)으로
  일관 RTF 목표 가능(선택).

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

# Changelog

## [0.3.0](https://github.com/groq/openbench/compare/v0.2.0...v0.3.0) (2025-08-14)


### Features

* add --debug flag to eval-retry command ([b26afaa](https://github.com/groq/openbench/commit/b26afaad31986e184c2695c6384cb1736ac0dfcb))
* add -M and -T flags for model and task arguments ([#75](https://github.com/groq/openbench/issues/75)) ([46a6ba6](https://github.com/groq/openbench/commit/46a6ba6b8a1d5a05b4ef1e53a9dcc1068967c4a8))
* add 'openbench' as alternative CLI entry point ([#48](https://github.com/groq/openbench/issues/48)) ([68b3c5b](https://github.com/groq/openbench/commit/68b3c5b4f8b8927dd5c6c8f68e25f831e9a5a222))
* add AI21 Labs inference provider ([#86](https://github.com/groq/openbench/issues/86)) ([db7bde7](https://github.com/groq/openbench/commit/db7bde7ea72eda2e688dd199d3e04e6505ccf1cc))
* add Baseten inference provider ([#79](https://github.com/groq/openbench/issues/79)) ([696e2aa](https://github.com/groq/openbench/commit/696e2aa760faf94db116405ebccb819e2ce6a2b5))
* add Cerebras and SambaNova model providers ([1c61f59](https://github.com/groq/openbench/commit/1c61f597ddc801caf3f085fa29fd35c50fed7b37))
* add Cohere inference provider ([#90](https://github.com/groq/openbench/issues/90)) ([8e6e838](https://github.com/groq/openbench/commit/8e6e838f447c7c0306c2c4f8523c7a9057046b0c))
* add Crusoe inference provider ([#84](https://github.com/groq/openbench/issues/84)) ([3d0c794](https://github.com/groq/openbench/commit/3d0c794dc5ef0d1eb188d3673e18f891850d0965))
* add DeepInfra inference provider ([#85](https://github.com/groq/openbench/issues/85)) ([6fedf53](https://github.com/groq/openbench/commit/6fedf53fa585fcaf9ff9a0bf396eab9a7c6a7f49))
* add Friendli inference provider ([#88](https://github.com/groq/openbench/issues/88)) ([7e2b258](https://github.com/groq/openbench/commit/7e2b25838e0c8725dbb8822099db826deabf2c8a))
* Add huggingface inference provider ([#54](https://github.com/groq/openbench/issues/54)) ([f479703](https://github.com/groq/openbench/commit/f479703a08f6605f70592d01a82588486650d49c))
* add Hyperbolic inference provider ([#80](https://github.com/groq/openbench/issues/80)) ([4ebf723](https://github.com/groq/openbench/commit/4ebf723c1577b542cef1c53f6bb254bc13c02a52))
* add initial GraphWalks benchmark implementation ([#58](https://github.com/groq/openbench/issues/58)) ([1aefd07](https://github.com/groq/openbench/commit/1aefd07befb8eeaebefd97066518e9d1a0523d73))
* add Lambda AI inference provider ([#81](https://github.com/groq/openbench/issues/81)) ([b78c346](https://github.com/groq/openbench/commit/b78c34690713c740af46d48eeedca967e15c64da))
* add MiniMax inference provider ([#87](https://github.com/groq/openbench/issues/87)) ([09fd27b](https://github.com/groq/openbench/commit/09fd27b4dfe043325c908bbce1aa00430259f2ee))
* add Moonshot inference provider ([#91](https://github.com/groq/openbench/issues/91)) ([e5743cb](https://github.com/groq/openbench/commit/e5743cbf4825c673d46ed98a157fee6e30961e6b))
* add Nebius model provider ([#47](https://github.com/groq/openbench/issues/47)) ([ba2ec19](https://github.com/groq/openbench/commit/ba2ec19ee1ac522133ed4dcd9b102d64a69933ff))
* add Nous Research model provider ([#49](https://github.com/groq/openbench/issues/49)) ([32dd815](https://github.com/groq/openbench/commit/32dd815002f9996c82bae001fdfc9b0ac7e09a0d))
* add Novita AI inference provider ([#82](https://github.com/groq/openbench/issues/82)) ([6f5874a](https://github.com/groq/openbench/commit/6f5874ae08891b9e6cae7160851114767b1f8fff))
* add Parasail inference provider ([#83](https://github.com/groq/openbench/issues/83)) ([973c7b3](https://github.com/groq/openbench/commit/973c7b32638144b6b766cec1af3eede3ac0b8743))
* add Reka inference provider ([#89](https://github.com/groq/openbench/issues/89)) ([1ab9c53](https://github.com/groq/openbench/commit/1ab9c536b9400177c8d8cdb827ae3b59a74991ff))
* add SciCode ([#63](https://github.com/groq/openbench/issues/63)) ([3650bfa](https://github.com/groq/openbench/commit/3650bfa7d87f729ac0288aca01df7c599894cb0b))
* add support for alpha benchmarks in evaluation commands ([#92](https://github.com/groq/openbench/issues/92)) ([e2ccfaa](https://github.com/groq/openbench/commit/e2ccfaa0faf934756094c7bf7be82e2f70c95059))
* push eval data to huggingface repo ([#65](https://github.com/groq/openbench/issues/65)) ([acc600f](https://github.com/groq/openbench/commit/acc600f4c567fe3a94154fd574a9b2c0a64b3762))


### Bug Fixes

* add missing newline at end of novita.py ([ef0fa4b](https://github.com/groq/openbench/commit/ef0fa4b4e16be82b3bb5238f0b06f28fb97c6537))
* remove default sampling parameters from CLI ([#72](https://github.com/groq/openbench/issues/72)) ([978638a](https://github.com/groq/openbench/commit/978638a274c67b1c84ca9c925438714cbeace175))


### Documentation

* docs for 0.3.0 ([#93](https://github.com/groq/openbench/issues/93)) ([fe358bb](https://github.com/groq/openbench/commit/fe358bbefdd6b2d30e444fe89ad2d45f8e07a1c1))
* fix directory structure documentation in CONTRIBUTING.md ([#78](https://github.com/groq/openbench/issues/78)) ([41f8ed9](https://github.com/groq/openbench/commit/41f8ed97c072306560dccaf96c1a55c973b6c708))


### Chores

* fix GraphWalks: Split into three separate benchmarks ([#76](https://github.com/groq/openbench/issues/76)) ([d1ed96e](https://github.com/groq/openbench/commit/d1ed96e3a8c45bd55e1b5a8b523063e13f6c7b06))
* update version ([8b7bbe7](https://github.com/groq/openbench/commit/8b7bbe74f14f67b2877cec3a6b3ae5e3a861a79a))


### Refactor

* move task loading from registry to config and update imports ([de6eea2](https://github.com/groq/openbench/commit/de6eea298d25be81be72b3c4986e72dd783c39cb))


### CI

* Enhance Claude code review workflow with updated prompts and model specification ([#71](https://github.com/groq/openbench/issues/71)) ([b605ed2](https://github.com/groq/openbench/commit/b605ed20528e8ddaa2da9107ef1808e46f0d91d1))

## [0.2.0](https://github.com/groq/openbench/compare/v0.1.1...v0.2.0) (2025-08-11)


### Features

* add DROP (simple-evals) ([#20](https://github.com/groq/openbench/issues/20)) ([f85bf19](https://github.com/groq/openbench/commit/f85bf194971f4a37b917d4d6ec6dfa31a1c3954c))
* add Humanity's Last Exam (HLE) benchmark ([#23](https://github.com/groq/openbench/issues/23)) ([6f10fb7](https://github.com/groq/openbench/commit/6f10fb71d6c8cabe8cddbb23bc0c979f8fb7234b))
* add MATH and MATH-500 benchmarks for mathematical problem solving ([#22](https://github.com/groq/openbench/issues/22)) ([9c6843b](https://github.com/groq/openbench/commit/9c6843babdfcbb85162cb88e71e3d2c71beeba5b))
* add MGSM ([#18](https://github.com/groq/openbench/issues/18)) ([bec1a7c](https://github.com/groq/openbench/commit/bec1a7c732912b235941e3cedfa1ff4f9092be0f))
* add openai MRCR benchmark for long context recall ([#24](https://github.com/groq/openbench/issues/24)) ([1b09ebd](https://github.com/groq/openbench/commit/1b09ebd13e765652ec1b6e8756599a28d9544224))
* HealthBench ([#16](https://github.com/groq/openbench/issues/16)) ([2caa47d](https://github.com/groq/openbench/commit/2caa47dad56faeaede219a41a0555d2887f782bc))


### Documentation

* update CLAUDE.md with pre-commit and dependency pinning requirements ([f33730e](https://github.com/groq/openbench/commit/f33730e570d55a2da171f0e44a0382bef749421e))


### Chores

* GitHub Terraform: Create/Update .github/workflows/stale.yaml [skip ci] ([1a00342](https://github.com/groq/openbench/commit/1a00342abde5d93dab3748157493a45dbf6a62b6))

## [0.1.1](https://github.com/groq/openbench/compare/v0.1.0...v0.1.1) (2025-07-31)


### Bug Fixes

* add missing __init__.py files and fix package discovery for PyPI ([#10](https://github.com/groq/openbench/issues/10)) ([29fcdf6](https://github.com/groq/openbench/commit/29fcdf6fefa48fcf480db1f84cf5845f7f7758ce))


### Documentation

* update README to streamline setup instructions for OpenBench, use pypi ([16e08a0](https://github.com/groq/openbench/commit/16e08a091b6fcc56422df21d1352bcc88481f175))

## 0.1.0 (2025-07-31)


### Features

* openbench ([3265bb0](https://github.com/groq/openbench/commit/3265bb07929f461a96d608d54fcdb144c66c0ac7))


### Chores

* **ci:** update release-please workflow to allow label management ([b70db16](https://github.com/groq/openbench/commit/b70db1665355be278af8a6d06f2a58aeedbe4a31))
* drop versions  for release ([58ce995](https://github.com/groq/openbench/commit/58ce9958b715c2f83fab509afdf046811b18c128))
* GitHub Terraform: Create/Update .github/workflows/stale.yaml [skip ci] ([555658a](https://github.com/groq/openbench/commit/555658af369b4e88eb92bf7f2afa2adcc4934835))
* update project metadata for version 0.1.0, add license, readme, and repository links ([9ea2102](https://github.com/groq/openbench/commit/9ea21029ebe3782d3d67b6aa075faf8862440fbf))

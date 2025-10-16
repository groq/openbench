# Changelog

## [0.5.1](https://github.com/groq/openbench/compare/v0.5.0...v0.5.1) (2025-10-16)


### Bug Fixes

* broken link ([#247](https://github.com/groq/openbench/issues/247)) ([5ef66e0](https://github.com/groq/openbench/commit/5ef66e0908652b4becf11b6d3a8b5497656f9c40))


### Chores

* update lockfile ([dff1bd9](https://github.com/groq/openbench/commit/dff1bd92e47bd9d4770da4a46f9ccfb3e90f83b2))


### Refactor

* extract cybersecurity benchmarks to plugin ([#251](https://github.com/groq/openbench/issues/251)) ([df829e2](https://github.com/groq/openbench/commit/df829e2482a481d4cde2f75d3cc906184eda330d))

## [0.5.0](https://github.com/groq/openbench/compare/v0.4.1...v0.5.0) (2025-10-10)


### ⚠ BREAKING CHANGES

* added more groupings under benchmarks catalog ([#244](https://github.com/groq/openbench/issues/244))

### Features

* add clockbench evaluation framwork and script for synthesizing public dataset. ([#159](https://github.com/groq/openbench/issues/159)) ([3ba9836](https://github.com/groq/openbench/commit/3ba98367252e4c0938d2841ff673662108cebe07))
* add IFEval ([#182](https://github.com/groq/openbench/issues/182)) ([8d1b939](https://github.com/groq/openbench/commit/8d1b939477a5a2202df4a874cb0fa0586fe2d9e3))
* add local openbench implementation of groq provider in inspect ([#131](https://github.com/groq/openbench/issues/131)) ([52aea35](https://github.com/groq/openbench/commit/52aea3510f03cd49d0fb66c3e6789f61939a7ee0))
* add mmmlu eval ([#193](https://github.com/groq/openbench/issues/193)) ([a42c2d5](https://github.com/groq/openbench/commit/a42c2d5499a87366cfbb62cc254d47021565a628))
* add mmstar benchmark ([#174](https://github.com/groq/openbench/issues/174)) ([5d085ab](https://github.com/groq/openbench/commit/5d085ab4b14a0a176953bfb3134e89e8eb36cb85))
* add new openbench documentation ([#169](https://github.com/groq/openbench/issues/169)) ([f3e6a37](https://github.com/groq/openbench/commit/f3e6a373c3195f90ed3017d81202ded81cde54d2))
* add overarching bbh command to run all 18 BBH tasks ([463a25f](https://github.com/groq/openbench/commit/463a25f308f8613f742f6bf79aba16d3998a30f3))
* add preset eval group infrastructure ([#215](https://github.com/groq/openbench/issues/215)) ([d9ea03a](https://github.com/groq/openbench/commit/d9ea03a76f1a580037cd9760aa5ee3505104e341))
* added more groupings under benchmarks catalog ([#244](https://github.com/groq/openbench/issues/244)) ([d932cb0](https://github.com/groq/openbench/commit/d932cb0b4aa8752a2075706f63485b3fb4dbdec0))
* **ArabicMMLU:** add remaining 32 Arabic exam subsets, total 41 subsets ([#219](https://github.com/groq/openbench/issues/219)) ([006e248](https://github.com/groq/openbench/commit/006e2480fd336642f5bbe0fc3a08b521fbbe7dc9))
* **benchmark:** add support for arc-agi ([#158](https://github.com/groq/openbench/issues/158)) ([3f32253](https://github.com/groq/openbench/commit/3f3225324e7578b7e6e1cbf3261fa73107b75745))
* **benchmark:** add support for detailbench ([#154](https://github.com/groq/openbench/issues/154)) ([23fbca5](https://github.com/groq/openbench/commit/23fbca5b09b4636b9c07886ce20e10ed1d386eb8))
* **benchmark:** add support for TUMLU ([#160](https://github.com/groq/openbench/issues/160))  ([#161](https://github.com/groq/openbench/issues/161)) ([885be75](https://github.com/groq/openbench/commit/885be75b68529e1b81aa964f6fbd0e0ea1de0ccd))
* **benchmark:** multichallenge implementation ([#170](https://github.com/groq/openbench/issues/170)) ([cf2ab4f](https://github.com/groq/openbench/commit/cf2ab4fcf77195a66360408f9b230cceae3732e8))
* change default model to groq/openai/gpt-oss-20b ([#138](https://github.com/groq/openbench/issues/138)) ([8f7f42f](https://github.com/groq/openbench/commit/8f7f42f1a3faba9dafa535f135a6a1638592ed4f))
* **components:** export the run_eval entrypoint method ([#157](https://github.com/groq/openbench/issues/157)) ([acbe7f4](https://github.com/groq/openbench/commit/acbe7f45c3174951c9f6232bd2a624ff2682121f))
* configure release-please for pre-v1.0 version bumping ([#133](https://github.com/groq/openbench/issues/133)) ([c432934](https://github.com/groq/openbench/commit/c4329343e059f6dd8c9a664a67431aba8d8fb853))
* **cybench:** ported over code for cybench ([#207](https://github.com/groq/openbench/issues/207)) ([7949425](https://github.com/groq/openbench/commit/794942597a248a454455acd33e56565807b77f48))
* cybersecurity, changelog, more docs ([39f123c](https://github.com/groq/openbench/commit/39f123cd4c0c98192a965827ab07256a8403d421))
* display results patch to include task duration stats ([#167](https://github.com/groq/openbench/issues/167)) ([e4e480c](https://github.com/groq/openbench/commit/e4e480cc9c73329e6fb9eaecdaa59b304a5db887))
* **docs:** add changelog page ([#225](https://github.com/groq/openbench/issues/225)) ([7db9135](https://github.com/groq/openbench/commit/7db913513cd6b2cefba948cafb11b65588b2eb30))
* **docs:** add release notes section and update index with new features for v0.5 ([#245](https://github.com/groq/openbench/issues/245)) ([09ab78e](https://github.com/groq/openbench/commit/09ab78ece5c6b85516503c0aaa682dff54eb260b))
* **docs:** added feature card and docs page for exercism ([#243](https://github.com/groq/openbench/issues/243)) ([2b38147](https://github.com/groq/openbench/commit/2b38147397814d6d27f51854784630f3e4ef94a8))
* **docs:** Added feature eval docs pages and cache command docs ([#191](https://github.com/groq/openbench/issues/191)) ([50501f1](https://github.com/groq/openbench/commit/50501f13d429ba52d82b55b21916aeac0fb2c972))
* **eval:** add support for json output ([#14](https://github.com/groq/openbench/issues/14)) ([f335418](https://github.com/groq/openbench/commit/f3354188fcbd35067072da8840a444188a3a0221))
* **exercism:** added support for exercism tasks w/ agent support for aider, roo, claude, opencode ([#151](https://github.com/groq/openbench/issues/151)) ([d86f0da](https://github.com/groq/openbench/commit/d86f0da6d98580a2ca6d4d1b63f5b89fd0a86175))
* graphwalks token filter ([#115](https://github.com/groq/openbench/issues/115)) ([e38658c](https://github.com/groq/openbench/commit/e38658c916d987bf8074910dfdefefb88e97faf6))
* groq reasoning effort + bugfix to override inspect's "groq" ([#142](https://github.com/groq/openbench/issues/142)) ([b919cc7](https://github.com/groq/openbench/commit/b919cc7e6bc748c92393a9a9bbf69db560313495))
* **lighteval:** Add 7 core commonsense reasoning benchmarks from LightEval ([#197](https://github.com/groq/openbench/issues/197)) ([7792c45](https://github.com/groq/openbench/commit/7792c45a2bb0aad825403fe7ccb14e0e8a537a9d))
* **lighteval:** add BigBench eval (122 MCQ tasks) ([9f35b1d](https://github.com/groq/openbench/commit/9f35b1daebf87d80666a4a291b12f18b616746be))
* **lighteval:** Add cross-lingual understanding benchmarks (XCOPA, XStoryCloze, XWinograd) ([917667a](https://github.com/groq/openbench/commit/917667ad8238a1319d64c0e9e0babf337c30f57d))
* **lighteval:** add Global-MMLU eval (42 languages) ([3542213](https://github.com/groq/openbench/commit/354221387cf5e3cf8754636810a0570ca64fe196))
* **lighteval:** register BigBench benchmarks in config and registry ([77018e9](https://github.com/groq/openbench/commit/77018e9c0b187b0d4a553f91054e9f1bc7089831))
* **lighteval:** register Global-MMLU benchmarks in config and registry ([156f509](https://github.com/groq/openbench/commit/156f509416d5a8f38214c9a20cf414cb4a6fc933))
* link to subscription form on main page ([#240](https://github.com/groq/openbench/issues/240)) ([988d08c](https://github.com/groq/openbench/commit/988d08c0d63a7c4246bb0123a45145c968ebeb44))
* **livemcpbench:** Adding support for liveMCPBench ([#127](https://github.com/groq/openbench/issues/127)) ([222f678](https://github.com/groq/openbench/commit/222f678369030a34144f8ac4fefba48835d7206f))
* make evals dash/undescore insensitive ([#185](https://github.com/groq/openbench/issues/185)) ([5ec5177](https://github.com/groq/openbench/commit/5ec5177ffa1c42d6f62a02f88226395914d4eba5))
* mbpp ([#117](https://github.com/groq/openbench/issues/117)) ([93ad88b](https://github.com/groq/openbench/commit/93ad88b56b6586405efef87a6a4a4e6c396c988c))
* **mcq_eval:** enable abstraction of MCQ eval ([#181](https://github.com/groq/openbench/issues/181)) ([2f53db2](https://github.com/groq/openbench/commit/2f53db20bb8d1474e0d49610c8e41c469dd07719))
* **mmmu-pro:** added support for mmmu_mcq, mmmu_open, mmmu_pro, mmmu_pro_vision ([#134](https://github.com/groq/openbench/issues/134)) ([a875378](https://github.com/groq/openbench/commit/a875378e62e3bbd3f8050d1a9685d3e05a47a53d))
* **openrouter:** add OpenRouter provider support ([#145](https://github.com/groq/openbench/issues/145)) ([47b579e](https://github.com/groq/openbench/commit/47b579e65c8eeb520023bc1e077ff50641f43bb6))
* **openrouter:** add provider routing args support ([#180](https://github.com/groq/openbench/issues/180)) ([12e1d81](https://github.com/groq/openbench/commit/12e1d818f9cecf2e74b930591981026d288557e2))
* **otis-mock-aime:** added support for otis mock aime 2024-2025 ([#218](https://github.com/groq/openbench/issues/218)) ([1b9fd5c](https://github.com/groq/openbench/commit/1b9fd5c8a65d8132fe1faf0e48d90d8b5f55bbfe))
* **plugins:** add entry point system for external benchmarks ([#216](https://github.com/groq/openbench/issues/216)) ([71e7257](https://github.com/groq/openbench/commit/71e7257298787905b05c0048ed9fbd6e3ff14169))
* return eval logs from run_eval function ([#173](https://github.com/groq/openbench/issues/173)) ([ee459d9](https://github.com/groq/openbench/commit/ee459d97bcb68039509b39b38783360e5c03c823))
* **rootly_terraform:** add initial implementation of Rootly Terraform evals ([#195](https://github.com/groq/openbench/issues/195)) ([cd3acae](https://github.com/groq/openbench/commit/cd3acaebf2a736942456f815f82c3bf54e2ac751))


### Bug Fixes

* allow for more python versions ([#164](https://github.com/groq/openbench/issues/164)) ([e6682fe](https://github.com/groq/openbench/commit/e6682fe60f2b69c5def4f0085f74006367fb6911))
* close headqa metadata entry ([947522d](https://github.com/groq/openbench/commit/947522dd2ef341c861cf1290687594b0765e2929))
* **cybench:** moved cybench dependency into dependency group ([#237](https://github.com/groq/openbench/issues/237)) ([8d30715](https://github.com/groq/openbench/commit/8d30715103dfe2476c8ebee0cb6b7158d8f0ad63))
* handle missing SciCode dependency lazily in solver ([#186](https://github.com/groq/openbench/issues/186)) ([fed4e88](https://github.com/groq/openbench/commit/fed4e888b7673a5e0d76d3c854ec1fa342fa040a))
* improve BBH target extraction to handle multi-char answers ([147e3e0](https://github.com/groq/openbench/commit/147e3e0696ad918d787ec82287cdd2c4b91498d2))
* load Neue Regrade font in Mintlify docs ([#177](https://github.com/groq/openbench/issues/177)) ([550c7f5](https://github.com/groq/openbench/commit/550c7f575c715a5c47aaf8609325070e0d788788))
* make core package actually install ([#235](https://github.com/groq/openbench/issues/235)) ([edeb4b8](https://github.com/groq/openbench/commit/edeb4b81134d9d138b8e9149f99f031288b05297))
* normalize benchmark keys during entry point merge ([#217](https://github.com/groq/openbench/issues/217)) ([d285664](https://github.com/groq/openbench/commit/d285664b40e893f792413810d67f550ea377907b))
* register headqa_en and headqa_es variants ([6a19aa1](https://github.com/groq/openbench/commit/6a19aa15b78570b5fb3bac715546e71d59daf1fe))
* render inspect error correctly ([#241](https://github.com/groq/openbench/issues/241)) ([97ccd10](https://github.com/groq/openbench/commit/97ccd106de80f5a6555d130c3bb7515c39de5fb3))
* resolve registry import conflict ([98a1c79](https://github.com/groq/openbench/commit/98a1c794436f06fc4297696530802125192ae27d))
* **scicode:** add support for test split, fix test_data.h5 import error ([#149](https://github.com/groq/openbench/issues/149)) ([23fa8cb](https://github.com/groq/openbench/commit/23fa8cbf4cc431ce4afdbe87b268afc7310e1b3f))
* update bbh function for programmatic access only ([811ce9e](https://github.com/groq/openbench/commit/811ce9ecb4f1ef0bb3cacce7fae8f298d0edfc61))
* use generic type ignore for bbh task decorator ([fd67171](https://github.com/groq/openbench/commit/fd67171e55a4901d523c517906ff06941410ce30))


### Documentation

* **readme:** clarify benchmark case-sensitivity and grader requirements ([#135](https://github.com/groq/openbench/issues/135)) ([c34a5a3](https://github.com/groq/openbench/commit/c34a5a362ec6764909258350f383ec561274b5bb))


### Chores

* add [@nmayorga7](https://github.com/nmayorga7) to CODEOWNERS ([abad7bf](https://github.com/groq/openbench/commit/abad7bf76e86319e98921b73455e2c06e72d2940))
* alphabetize available benchmarks error ([#214](https://github.com/groq/openbench/issues/214)) ([68f46e9](https://github.com/groq/openbench/commit/68f46e9304994d67b720e6dae40e328292f8ed57))
* **benchmark:** removed combined cti-bench eval ([#183](https://github.com/groq/openbench/issues/183)) ([a77852c](https://github.com/groq/openbench/commit/a77852c69c210ed10ba410bd769c21a94789f5f7))
* bugbot fixes for MCQ ([#190](https://github.com/groq/openbench/issues/190)) ([6ecaefc](https://github.com/groq/openbench/commit/6ecaefcafdd471ec0b9e2949d1ec3ee448da0447))
* **docs:** add docs for openrouter and MCQEval ([#188](https://github.com/groq/openbench/issues/188)) ([7f8cd83](https://github.com/groq/openbench/commit/7f8cd83909f13b260078501fbac72cbeb60e0f97))
* **docs:** alphabetize benchmarks metadata ([#187](https://github.com/groq/openbench/issues/187)) ([ce77812](https://github.com/groq/openbench/commit/ce778121a700f423132809dbf371468b45f150b2))
* **docs:** benchmarks each on new line ([#184](https://github.com/groq/openbench/issues/184)) ([b3c40f8](https://github.com/groq/openbench/commit/b3c40f82fb5b26c0aba2ff0a17c9af45222f177d))
* **docs:** minor cleanup ([#179](https://github.com/groq/openbench/issues/179)) ([80c9e09](https://github.com/groq/openbench/commit/80c9e096885817b7a3dc62220f365760347aa462))
* fixed fonts in openbench docs ([#178](https://github.com/groq/openbench/issues/178)) ([6e3c2a5](https://github.com/groq/openbench/commit/6e3c2a5405e54b274656be34a9385d9b307fcd38))
* GitHub Terraform: Create/Update .github/workflows/stale.yaml [skip ci] ([d0018d1](https://github.com/groq/openbench/commit/d0018d1cac7b024844b530117b31a09088882728))
* **mcq-eval:** accept more dataset types ([#194](https://github.com/groq/openbench/issues/194)) ([cb5e038](https://github.com/groq/openbench/commit/cb5e038578ded316582f82c8d65503e7146e7781))
* move all metrics to discrete files in /metrics ([#168](https://github.com/groq/openbench/issues/168)) ([7caa042](https://github.com/groq/openbench/commit/7caa042318a8ac27b496357c64b2489bd0da6d43))
* release-please pre-1.0: treat BREAKING as minor ([3c88a42](https://github.com/groq/openbench/commit/3c88a429573ef81889751e2b3013a70e07b594f1))
* remove pre-commit benchmark checks for easier CI ([#213](https://github.com/groq/openbench/issues/213)) ([67b07a7](https://github.com/groq/openbench/commit/67b07a7954e6bf6b87966f7347b65323aa159f3e))
* rename OpenBench to openbench ([#196](https://github.com/groq/openbench/issues/196)) ([0621b46](https://github.com/groq/openbench/commit/0621b4691eb1f08c3110fada527752fc95e72add))
* rename task to sample in time metrics ([#172](https://github.com/groq/openbench/issues/172)) ([96e1817](https://github.com/groq/openbench/commit/96e1817d79ab24a01c5680dc31a7dcab01e11d85))
* sync packaging pyproject ([#234](https://github.com/groq/openbench/issues/234)) ([940a879](https://github.com/groq/openbench/commit/940a879f9bba7939c1920065a4593cb899b93999))
* update Claude workflows to enhance permissions and streamline triggers ([#136](https://github.com/groq/openbench/issues/136)) ([effb7da](https://github.com/groq/openbench/commit/effb7da7ce12ba11da311bfb52493fb8b40dac24))
* update readme and contributing ([#176](https://github.com/groq/openbench/issues/176)) ([ea606ba](https://github.com/groq/openbench/commit/ea606ba82ce0a385063288c6b06616e9d1e9f84c))
* update release-please configuration and add lockfile update workflow ([#146](https://github.com/groq/openbench/issues/146)) ([2d6ad9b](https://github.com/groq/openbench/commit/2d6ad9b529e7092c554dabee9f479d6555a698c7))
* user agent ([#163](https://github.com/groq/openbench/issues/163)) ([e20f3c1](https://github.com/groq/openbench/commit/e20f3c1bf9af74c14cee26f3dc92d0e85cbfed0a))


### CI

* add benchmarks validation pre commit hook ([#171](https://github.com/groq/openbench/issues/171)) ([3725638](https://github.com/groq/openbench/commit/37256380e95dce060b8a8ceb48d719356d7b95b3))
* remove PR trigger from release-please ([#166](https://github.com/groq/openbench/issues/166)) ([6440d44](https://github.com/groq/openbench/commit/6440d44ba0a13834c6d77d913861c132de4482d1))

## [0.4.1](https://github.com/groq/openbench/compare/v0.4.0...v0.4.1) (2025-08-29)


### Bug Fixes

* **rootly_gmcq:** handle both string and list content types in scorer ([#129](https://github.com/groq/openbench/issues/129)) ([376624d](https://github.com/groq/openbench/commit/376624d1c24ce876dc6bbd1a3c09b566e7b303b5))

## [0.4.0](https://github.com/groq/openbench/compare/v0.3.0...v0.4.0) (2025-08-28)


### Features

* add boolq ([#70](https://github.com/groq/openbench/issues/70)) ([edbd1cc](https://github.com/groq/openbench/commit/edbd1cc1227e83a4de2d1c383acbc2c914063018))
* add BrowseComp ([#118](https://github.com/groq/openbench/issues/118)) ([498c706](https://github.com/groq/openbench/commit/498c7063f67f1ae5f2b68420269ad939e4a684ab))
* add CITATION.cff for software citation ([#102](https://github.com/groq/openbench/issues/102)) ([16960de](https://github.com/groq/openbench/commit/16960dec7b75d55de8f60d78cee99a691a85d083))
* add CTI-Bench cybersecurity benchmark suite ([#96](https://github.com/groq/openbench/issues/96)) ([8465075](https://github.com/groq/openbench/commit/84650753e3b7b2cbe35b150d6b6466985d07e01d))
* add GitHub issue and PR templates ([#103](https://github.com/groq/openbench/issues/103)) ([68f0ef0](https://github.com/groq/openbench/commit/68f0ef0a514cabaa165ba343ca38d87edebe4452))
* add gmcq ([#114](https://github.com/groq/openbench/issues/114)) ([bb3c89d](https://github.com/groq/openbench/commit/bb3c89d03a6baf1dcc5e608a7438068e7f2f3d35))
* add MuSR variants and grouped metrics ([#107](https://github.com/groq/openbench/issues/107)) ([10ae935](https://github.com/groq/openbench/commit/10ae935254531813b3dc087a7f127c08fad3422e))
* add robust answer extraction scorers from gpt-oss to MathArena benchmarks and gpqa_diamond ([#97](https://github.com/groq/openbench/issues/97)) ([251ba66](https://github.com/groq/openbench/commit/251ba66b5e65cb30f1bd0afaaf1ac4a96e75a0ad))
* add Vercel AI Gateway inference provider ([#98](https://github.com/groq/openbench/issues/98)) ([38e211a](https://github.com/groq/openbench/commit/38e211ab0cfa042b80d7bc62e02b529a816eb090))
* jsonschemabench ([#95](https://github.com/groq/openbench/issues/95)) ([e3d842d](https://github.com/groq/openbench/commit/e3d842d10ee824baa882d8cb9e1f7c3e4adf28e2))
* **mmmu:** added support for mmmu benchmark and all of its subdomains ([#121](https://github.com/groq/openbench/issues/121)) ([801bceb](https://github.com/groq/openbench/commit/801bcebe9e92b71022440a5c5788ac8b377a762e))


### Bug Fixes

* format mmlu_pro.py dataset file ([2a9ee65](https://github.com/groq/openbench/commit/2a9ee651f680da10c3e5a2403d9103821e9e52bc))
* handle skipped integration tests in CI ([#120](https://github.com/groq/openbench/issues/120)) ([dae9378](https://github.com/groq/openbench/commit/dae937838b90ba39fb134daf694ea4bc3563508c))
* **hle:** added multimodal support for hle ([#128](https://github.com/groq/openbench/issues/128)) ([8c3f212](https://github.com/groq/openbench/commit/8c3f212b7cba8650a47f4bb297213265b4fac660))
* **jsonschemaeval:** match paper methodology and add openai subset ([#113](https://github.com/groq/openbench/issues/113)) ([1b6470b](https://github.com/groq/openbench/commit/1b6470b3671f76eb809b8af55593ad9529179546))
* make claude-code-review job optional to prevent PR blocking ([#100](https://github.com/groq/openbench/issues/100)) ([6aad080](https://github.com/groq/openbench/commit/6aad080efc7fb2033e760248342d8f18e8f0709e))


### Documentation

* emphasize pre-commit hooks installation requirement ([#106](https://github.com/groq/openbench/issues/106)) ([e765464](https://github.com/groq/openbench/commit/e7654641274eb60f07fd74e5db211bd7a3cca429))
* refresh CONTRIBUTING.md and update README references ([#105](https://github.com/groq/openbench/issues/105)) ([bf66747](https://github.com/groq/openbench/commit/bf66747481681ebd4f3615b96e9af46a16106836))
* update installation instructions and clarify dependency architecture in CLAUDE.md and CONTRIBUTING.md ([#126](https://github.com/groq/openbench/issues/126)) ([cd962fd](https://github.com/groq/openbench/commit/cd962fd19ee06c2b4a9c1f0e163c55c439567f7a))
* update README citation to match CITATION.cff ([#104](https://github.com/groq/openbench/issues/104)) ([6219e8c](https://github.com/groq/openbench/commit/6219e8c795ebddffd7a8086db009e6085e2ffe22))


### Chores

* bump Inspect-AI to 0.3.125 ([#124](https://github.com/groq/openbench/issues/124)) ([d728cbb](https://github.com/groq/openbench/commit/d728cbb6e777a64f923dfe910e0a5ca2d00a67d2))
* unpin dependencies except inspect-ai ([#108](https://github.com/groq/openbench/issues/108)) ([50cf90f](https://github.com/groq/openbench/commit/50cf90ff96bbc163a4040aaf632b729f4aaed7c0))
* update uv.lock package version ([3583d71](https://github.com/groq/openbench/commit/3583d71fee81d681861e0015383f8471459b4002))


### CI

* add automated PyPI publishing to release workflow ([#99](https://github.com/groq/openbench/issues/99)) ([eddbf70](https://github.com/groq/openbench/commit/eddbf70bbf13383abf6de633989a0a2a6ac8b593))

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

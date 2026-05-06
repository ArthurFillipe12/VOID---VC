# Projeto VOID---VC

Projeto acadêmico de Visão Computacional baseado no artigo **VOID: Video Object and Interaction Deletion** (Netflix Research), com foco em **melhoria de implementação**.

## Objetivo do projeto

Implementar e validar um módulo de **avaliação temporal automática** para o framework VOID, reduzindo a dependência de avaliação apenas visual/subjetiva.

Métricas implementadas:
- LPIPS temporal (frame a frame)
- Optical Flow Consistency (erro de warp com RAFT)
- PSNR consecutivo
- SSIM consecutivo

Saídas produzidas:
- JSON com resumo agregado
- CSV com métricas por par de frames

## Estrutura do repositório

- `notebooks/`: notebooks usados no projeto
- `docs/`: documentação textual de proposta, metodologia, resultados e limitações
- `article/`: artigo em LaTeX no formato solicitado pela disciplina
- `outputs/`: resultados exportados (JSON/CSV e imagens de apoio)
- `videox_fun/utils/temporal_metrics.py`: módulo de métricas temporais
- `tests/test_temporal_metrics.py`: teste unitário do módulo

## Notebooks

- `notebooks/00_void_original.ipynb`: notebook original do repositório
- `notebooks/01_void_framework_integration.ipynb`: execução e integração no VOID (inclui limitações de hardware)
- `notebooks/02_temporal_metrics_validation.ipynb`: validação isolada e reproduzível do módulo de métricas

## Resultados principais (validação do módulo)

Execução em vídeo real de exemplo (`sample/lime/input_video.mp4`), 8 frames, 7 pares:

- `lpips_temporal_mean`: `0.018937`
- `optical_flow_consistency_l1_mean`: `0.005620`
- `psnr_consecutive_mean`: `28.631974`
- `ssim_consecutive_mean`: `0.925345`

## Limitações encontradas

No Colab gratuito (Tesla T4 ~14.56 GB VRAM), a execução completa da inferência pesada do VOID pode falhar por memória (`exit code 137`).

Mesmo com essa limitação, a melhoria proposta foi:
- implementada no código
- integrada aos scripts
- validada funcionalmente com geração de artefatos quantitativos

## Como reproduzir rapidamente

1. Clonar este repositório.
2. Instalar dependências (`requirements.txt` + `lpips`).
3. Executar `notebooks/02_temporal_metrics_validation.ipynb` para validação direta do módulo.
4. Conferir arquivos gerados em `outputs/`.

## Referências

- S. Motamed et al., "VOID: Video Object and Interaction Deletion", arXiv.
- Repositório original VOID: https://github.com/Netflix/void-model
- Z. Teed and J. Deng, "RAFT", ECCV 2020.
- R. Zhang et al., "LPIPS", CVPR 2018.
- Z. Wang et al., "SSIM", IEEE TIP 2004.

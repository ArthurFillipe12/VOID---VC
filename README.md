# Projeto VOID---VC

Projeto acadêmico da disciplina de Visão Computacional (UFRPE) baseado no framework VOID (Netflix Research), com foco em uma melhoria de implementação para avaliação temporal automática.

## Escopo da melhoria

Foi implementado um módulo de métricas temporais para reduzir dependência de avaliação exclusivamente visual/subjetiva na remoção de objetos em vídeo.

Métricas calculadas por pares consecutivos de frames:
- LPIPS temporal
- Optical Flow Consistency (L1)
- PSNR consecutivo
- SSIM consecutivo

Saídas geradas:
- JSON com série temporal e resumo agregado
- CSV com métricas por par de frames

## Estrutura do repositório

- `config/`: arquivos de configuração do pipeline
- `docs/`: documentação textual do projeto
- `inference/`: scripts de inferência
- `notebooks/`: caderno base utilizado no ambiente Colab
- `sample/`: vídeos de exemplo
- `tests/test_temporal_metrics.py`: teste unitário da melhoria
- `videox_fun/utils/temporal_metrics.py`: implementação do módulo temporal
- `article/main.tex`: artigo final em LaTeX (formato IEEE)

## Resultado de validação funcional

Validação no clipe `sample/lime/input_video.mp4` com 8 frames (7 pares):

- `lpips_temporal_mean`: `0.018937`
- `optical_flow_consistency_l1_mean`: `0.005620`
- `psnr_consecutive_mean`: `28.631974`
- `ssim_consecutive_mean`: `0.925345`

## Limitação conhecida

No Colab gratuito (Tesla T4, ~14.56 GB VRAM), a inferência completa do pipeline pode encerrar por memória (`exit code 137`).

Essa limitação não impede a entrega da melhoria proposta, que foi implementada, integrada e validada funcionalmente.

## Reprodução rápida

1. Clonar o repositório.
2. Instalar dependências de `requirements.txt`.
3. Executar o fluxo de validação no Colab.
4. Conferir os artefatos JSON/CSV gerados.

## Entrega da disciplina

Para submissão, os itens essenciais deste repositório são:
- Código da melhoria (`videox_fun/utils/temporal_metrics.py`)
- Evidências de validação (`outputs/` quando disponível)
- Documentação (`docs/`)
- Artigo (`article/main.tex` e PDF final exportado)

## Referências

- S. Motamed et al., "VOID: Video Object and Interaction Deletion", arXiv.
- Repositório original VOID: https://github.com/Netflix/void-model
- Z. Teed and J. Deng, "RAFT", ECCV 2020.
- R. Zhang et al., "LPIPS", CVPR 2018.
- Z. Wang et al., "SSIM", IEEE TIP 2004.

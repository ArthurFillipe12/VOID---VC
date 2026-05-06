# Limitações

Durante os testes no Colab gratuito (Tesla T4 ~14.56 GB VRAM), a execução completa da inferência do VOID apresentou encerramento por memória (`exit code 137`).

Impacto:
- Não foi possível concluir, no mesmo ambiente, toda a comparação pesada Pass1 vs Pass2.
- A implementação e validação do módulo de métricas permaneceram viáveis e reproduzíveis.

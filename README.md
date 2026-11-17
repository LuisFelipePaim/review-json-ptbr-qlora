# Review JSON PT-BR com QLoRA

Projeto de portfólio para extração estruturada de avaliações de restaurantes e
delivery em português brasileiro. A entrada será uma avaliação livre; a saída,
um JSON com `sentimento`, `categorias`, `resumo` e `requer_ação`.

## Estado do projeto

Etapa 1: ambiente, estrutura e inferência sintética em 4 bits.
Ainda não há treino, baseline experimental, dataset derivado ou resultados de
qualidade. Cada etapa seguinte depende da confirmação do resultado da anterior.

Decisões aprovadas:

- Modelo: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), Apache-2.0.
- Dados propostos: [StarRest v2](https://data.mendeley.com/datasets/r5th4s7vpv/2),
  CC BY 4.0, sujeito à auditoria de idioma e cobertura das categorias de delivery.
- Rotulagem futura: professor aberto maior no Colab e revisão humana; cada
  exemplo terá origem e estado de revisão explícitos. Parte dos rótulos será sintética.
- Teste ouro: 250 avaliações separadas antes de qualquer treino, revisadas à mão
  e nunca usadas na escolha de prompts, hiperparâmetros ou checkpoints.

## Etapa 1 — Windows com Python 3.10

O perfil de referência usa Windows x86-64, GPU NVIDIA com suporte a NF4,
Python 3.10 e wheels oficiais. O instalador cria `.venv` isolado. Não é
necessário ativá-lo nem modificar a política global de execução do PowerShell.

```powershell
cd 'C:\Users\Luis Felipe\Downloads\review-json-ptbr-qlora'
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Versões diretas estão em `requirements.txt`. A wheel CUDA do PyTorch vem do
índice oficial `https://download.pytorch.org/whl/cu121`; o runtime vem na wheel,
portanto este caminho não exige compilar bitsandbytes nem instalar CUDA Toolkit
separadamente. O driver NVIDIA precisa ser compatível.

Quando a wheel já está em `.cache/wheels/`, o instalador confere seu SHA-256
contra o hash oficial fixado no script e reutiliza o arquivo. O perfil de
referência usa PyTorch 2.5.1 + CUDA 12.1; os ambientes globais são preservados.

Após a primeira validação, `requirements-windows.lock.txt` registra também as
dependências transitivas. Para reproduzir esse ambiente, acrescente `-UseLock`
ao comando do instalador. O lock é deste perfil Windows; o Colab terá seu próprio
perfil de dependências quando necessário.

Execute o diagnóstico:

```powershell
.\.venv\Scripts\python.exe -m review_json_ptbr.runtime
```

Saída esperada: `status: "ok"`, `cuda_available: true`, identificação da GPU,
versões dos pacotes, memória disponível e `nf4_cuda_forward.status: "ok"`.
Um forward de camada NF4 pequena confirma o kernel CUDA antes do download do
LLM, sem backward ou otimizador. Relatório em
`reports/local/environment.json`; indisponibilidade de CUDA retorna código 1.

Execute a inferência curta:

```powershell
.\.venv\Scripts\python.exe -m review_json_ptbr.inference.smoke
```

Na primeira execução haverá download dos pesos originais, aproximadamente
4,1 GB nesta revisão (4.063.479.808 bytes segundo o índice dos pesos), para o
cache do Hugging Face. Quantização é aplicada ao carregar na GPU.
O SHA do checkpoint está fixado em `configs/smoke_qwen3_1_7b.json`.
O cliente `hf-xet` está explicitamente instalado também no Windows para o
download pelo transporte Xet do Hub.

Saída esperada: `status: "ok"`, `is_loaded_in_4bit: true`, número positivo de
camadas `Linear4bit`, resposta curta e memória de pico. O relatório completo
fica em `reports/local/smoke_qwen3_1_7b.json`, inclusive em caso de erro.
Todos os parâmetros devem estar na GPU; não há fallback silencioso para CPU.

O texto usado é sintético: “O pedido chegou atrasado e a comida estava fria.”
O teste não acessa StarRest, não treina adaptadores e não serve como baseline
de qualidade. A prévia de leitura do JSON apenas informa se `json.loads`
aceitou a saída: a validação do esquema será desenvolvida na etapa 3.
Uma geração sem warmup também não é um benchmark de latência. Inferência
funcionar não demonstra que um passo de treino com otimizador cabe em 6 GB.
Os picos registrados nesta etapa são do alocador do PyTorch neste processo;
não incluem toda a memória do driver ou de outras aplicações.

Verificações locais:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src
```

## Conceitos usados nesta etapa

- **Chat template:** o tokenizer formata mensagens e marcadores de papéis no
  formato esperado pelo checkpoint. Mantemos o template oficial e desativamos
  raciocínio explícito (`enable_thinking=False`).
- **Tokenização:** converte o texto formatado em IDs. Não duplicamos tokens
  especiais depois de aplicar o template; não truncamos o prompt deste teste.
- **4 bits:** NF4 com dupla quantização comprime pesos das camadas elegíveis.
  Parte do modelo permanece em precisão maior; ativações e cache também ocupam
  memória. O cálculo deste perfil usa FP16.
- **Geração:** `do_sample=False` usa decodificação gulosa; o limite de novos
  tokens evita geração ilimitada. Temperatura e top-p não controlam esse modo.

LoRA, mascaramento da loss e otimização serão introduzidos nas etapas de treino,
após separar e revisar os dados.

## Estrutura

```text
configs/                    configuração versionada do smoke test
scripts/                    instalação Windows
src/review_json_ptbr/
  runtime.py                diagnóstico do ambiente
  inference/                geração sintética em NF4
  data/                     futura ingestão e separação
  prompts/                  futuros prompts versionados
  training/                 futuro BERTimbau e QLoRA
  evaluation/               futuro parser, validador e métricas
  benchmark/                futuras medições com warmup e repetições
tests/                      futuros testes unitários do contrato e métricas
reports/local/              relatórios locais, ignorados pelo Git
requirements.txt            versões diretas
```

## Etapas seguintes

2. Auditar StarRest, deduplicar e congelar os manifestos dos splits e do ouro.
3. Definir contrato de anotação, parser/validador, métricas e seus testes; rotular.
4. Medir zero-shot, few-shot e BERTimbau para sentimento.
5. Validar viabilidade de treino QLoRA, máscara da loss e consumo de memória.
6. Escolher hiperparâmetros na validação e executar ranks 4/8/16/32 com três seeds.
7. Comparar adaptador NF4, mesclagem e variantes em GPU e CPU.
8. Congelar configurações finais, avaliar uma vez no ouro e analisar erros.
9. Gerar gráficos dos CSVs e publicar código, adaptador, dados permitidos e Space.

Cada execução experimental terá configuração, hashes, seed e métricas em CSV;
média e desvio padrão virão das execuções reais. BERTimbau será comparado no
sentimento; campos generativos serão não aplicáveis para esse baseline.

## Fontes de compatibilidade

- [bitsandbytes: instalação e Windows](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation)
- [Transformers 4.57.6 e Python suportado](https://pypi.org/project/transformers/4.57.6/)
- [PEFT 0.18.1](https://pypi.org/project/peft/0.18.1/)
- [Preparação para treino quantizado](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Template de chat](https://huggingface.co/docs/transformers/v4.57.1/chat_templating)

Código, pesos e dados têm origens e licenças distintas. O projeto ainda não
publica pesos nem dados e não fez avaliação de generalização para delivery.


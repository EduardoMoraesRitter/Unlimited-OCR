# Exemplos: OCR para JSON

Estes exemplos preservam a resposta original do Unlimited-OCR e a convertem
deterministicamente para JSON. Pedir "responda em JSON" ao modelo não é
necessário e pode produzir JSON inválido; a conversão é feita por
`unlimited_ocr_json.py` depois da inferência.

Execute os comandos a partir da raiz do repositório.

Escolha o fluxo pelo que você quer comprovar:

| Fluxo | Executa o modelo? | Precisa de CUDA? | Resultado principal |
| --- | --- | --- | --- |
| Conversão de resposta salva | Não | Não | JSON determinístico a partir de texto bruto já existente |
| Imagem pela CLI | Sim | Sim | Texto bruto e JSON de uma imagem |
| Interface local | Sim, ao clicar em executar | Sim | Prévia visual, status CUDA, texto bruto e JSON para download |
| PDF difícil página a página | Sim | Sim | Um JSON agregado, respostas brutas por página e metadados de tentativas |

## 1. Converter uma resposta salva, sem GPU

O arquivo [`raw_output.txt`](raw_output.txt) simula um relatório em português
com acentos, caixas de detecção e uma tabela HTML. Este comando não usa rede,
modelo ou GPU:

```powershell
py examples\parse_output_to_json.py
```

No WSL/Linux:

```bash
python examples/parse_output_to_json.py
```

O resultado padrão é `outputs/raw_output.json`. Para converter outro arquivo:

```powershell
py examples\parse_output_to_json.py C:\docs\resposta.txt --output outputs\resposta.json
```

Para validar o conversor e os auxiliares sem baixar o modelo:

```powershell
py -3.12 -m pip install requests Pillow==12.1.1 pymupdf==1.27.2.2
py -3.12 -m unittest discover -s tests -v
```

São 24 testes offline: 18 cobrem parser, segurança e orquestração, e 6 cobrem
controles, limites e prévia real de imagem/PDF. Pillow e PyMuPDF são necessários
para os testes visuais, mas CUDA, pesos e Transformers não são.

## 2. Imagem única com Transformers

Esta etapa executa o modelo de OCR de verdade. Diferentemente do conversor da
seção anterior, ela precisa dos pesos do modelo e de PyTorch com CUDA. Essa
dependência vem do fluxo Transformers original do Unlimited-OCR, que chama
`model.eval().cuda()`; não foi adicionada por este fork.

O ambiente documentado usa Python 3.12, CUDA 12.9, PyTorch 2.10.0,
Transformers 4.57.1 e Pillow 12.1.1. Para reproduzir exatamente essa pilha em
uma máquina Windows com NVIDIA, use Ubuntu no WSL2. Na validação deste guia, o
índice CUDA 12.9 do PyTorch oferecia os wheels 2.10.0 para Linux, mas não para
Windows nativo; instalar apenas `torch==2.10.0` pelo PyPI no Windows resultava
na edição CPU, incompatível com estes exemplos.

WSL2/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install transformers==4.57.1 Pillow==12.1.1 \
  matplotlib==3.10.8 einops==0.8.2 addict==2.4.0 \
  easydict==1.13 pymupdf==1.27.2.2 psutil==7.2.2
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
python examples/transformers_image_to_json.py /mnt/c/docs/pagina.png \
  --mode base \
  --max-length 4096 \
  --revision 07dea832e22aefee32ad281d4b80551282e1c168
```

A verificação precisa terminar com `True`. Se mostrar `False`, não prossiga com
o download dos pesos: o Python está usando uma edição sem CUDA. O Windows nativo
continua adequado para o conversor JSON e para os 24 testes offline, que não
carregam o modelo.

O script chama `model.infer(..., eval_mode=True)`. Por padrão, grava
`outputs/pagina.raw.txt` e `outputs/pagina.json`. Use `--output` e
`--raw-output` para escolher outros destinos.

Modos disponíveis:

- `base`: uma visão global de 1024 × 1024; consome menos memória.
- `gundam`: visão global e recortes de 640 × 640; pode recuperar letras
  pequenas, mas usa mais VRAM e demora mais.

Para repetir os dois testes com imagens já versionadas no repositório:

```bash
python examples/transformers_image_to_json.py assets/baidu.png \
  --mode base --max-length 4096 \
  --revision 07dea832e22aefee32ad281d4b80551282e1c168 \
  --output outputs/repro/baidu.json \
  --raw-output outputs/repro/baidu.raw.txt

python examples/transformers_image_to_json.py assets/Unlimited-OCR.png \
  --mode base --max-length 4096 \
  --revision 07dea832e22aefee32ad281d4b80551282e1c168 \
  --output outputs/repro/overview.json \
  --raw-output outputs/repro/overview.raw.txt
```

O commit do modelo é fixado para reduzir variações do código remoto. O primeiro
uso ainda baixa aproximadamente 6,3 GB de pesos/cache do Hugging Face; execuções
seguintes normalmente reutilizam esse cache.

## 3. Interface web local, estilo Hugging Face

Depois de concluir a instalação CUDA da seção anterior, instale a combinação
testada da interface sem alterar as versões do Transformers e do Pillow:

```bash
python -m pip install gradio==6.15.1 accelerate==1.14.0 huggingface-hub==0.36.0
python examples/gradio_cuda_app.py
```

Abra `http://127.0.0.1:7860`. A tela aceita PNG, JPEG, WebP, BMP ou PDF, mostra
a GPU CUDA detectada, preserva o texto bruto, exibe o JSON e oferece o arquivo
`.json` para download. O CSS personalizado usa fontes do próprio sistema e não
depende de uma webfont externa.

### Prévia visual de imagens e PDFs

A prévia é atualizada automaticamente quando o arquivo muda. Para PDF, ela
também é renderizada novamente quando a página ou o DPI muda, antes de executar
o modelo. O painel informa a página selecionada, o total de páginas e a
resolução em pixels que será enviada ao OCR. Assim é possível confirmar
visualmente se a página certa está legível, girada ou cortada sem consumir VRAM
com uma inferência. Para imagens, a própria imagem enviada aparece no mesmo
painel.

A prévia do navegador é deliberadamente mais leve: usa no máximo 120 DPI, 1.600
pixels por lado e 3 milhões de pixels. O painel informa tanto o tamanho dessa
prévia quanto a resolução integral preparada para o OCR. Ao trocar o arquivo, a
página volta para 1; em PDFs o seletor recebe o total real de páginas e fica
ativo, enquanto em imagens ele deixa de ser necessário. Trocar página ou DPI
atualiza imagem e metadados automaticamente.

Os exemplos rápidos usam arquivos públicos já presentes no repositório:

| Exemplo | O que observar |
| --- | --- |
| [`assets/baidu.png`](../assets/baidu.png) | Smoke test mínimo, com o texto curto `Baidu 百度`. |
| [`assets/Unlimited-OCR.png`](../assets/Unlimited-OCR.png) | Diagrama com texto e elementos gráficos; útil para revelar que uma execução tecnicamente válida pode não recuperar conteúdo. |
| [`Unlimited-OCR.pdf`](../Unlimited-OCR.pdf) | PDF de 14 páginas do próprio projeto; escolha uma página e confira a prévia antes de executar. |

Como verificação offline da interface, a primeira página desse PDF em 200 DPI
informa entrada OCR de 1654 × 2339 pixels, e `assets/baidu.png` mantém sua prévia
original de 440 × 133. Esses testes de renderização não carregam o modelo.

Passo a passo:

1. Selecione um exemplo ou envie um arquivo.
2. Em PDF, escolha a página e o DPI e confira a prévia, o total de páginas e a
   resolução renderizada. Em imagem, confira a própria imagem.
3. A interface fixa o modo `base`; mantenha o limite em no máximo 4.096 nessa
   GPU de 8 GB.
4. Clique em **Executar OCR na GPU** e aguarde o status final.
5. Compare a aba de texto bruto com a imagem, inspecione avisos no JSON e faça o
   download do `.json`. O status repete o arquivo, a página e o DPI efetivamente
   processados, mesmo que você altere a prévia durante uma execução longa.
6. Clique em **Liberar GPU** quando terminar.

O modelo só é carregado no primeiro clique em **Executar OCR na GPU**. As
requisições são serializadas e usam apenas o modo `base`, adequado ao limite de
8 GB da RTX 4070 testada. A interface limita a sequência a 4.096, o upload a 25
MB e a imagem renderizada a 25 milhões de pixels. O botão **Liberar GPU** remove
os pesos sem fechar a página. O servidor usa apenas `127.0.0.1` e
`share=False`: ele não cria uma URL pública nem envia o documento para um
Gradio Space.

Cada execução fica em `outputs/web-ui/<data-hora>/`, com a resposta `.raw.txt`
e o JSON. Essa pasta é ignorada pelo Git. O JSON da interface usa somente o nome
do arquivo e a página como origem, em vez do caminho absoluto do computador, e
mantém o texto bruto no arquivo separado.

Se a página não abrir, confirme que o processo ainda está ativo e que a porta
7860 está livre. Para encerrar, use `Ctrl+C` no terminal. Não use
`share=True` com documentos privados.

## 4. Resultados CUDA reais de 2026-08-06

O ambiente observado foi Ubuntu 24.04 no WSL2, Python 3.12, PyTorch
2.10.0+cu129, CUDA 12.9, Transformers 4.57.1 e NVIDIA GeForce RTX 4070 Laptop
GPU. Os três testes usaram o modelo `baidu/Unlimited-OCR` na revisão
`07dea832e22aefee32ad281d4b80551282e1c168`, modo `base` e
`max_length=4096`.

| Entrada | Caminho testado | Tempo de parede | Pico observado | Estrutura produzida | Resultado de qualidade |
| --- | --- | ---: | ---: | --- | --- |
| `assets/baidu.png` (440 × 133) | Interface local | 92 s | 6.835 MiB de alocação CUDA pelo PyTorch | JSON 1.0, 1 página, 1 bloco `title`, sem avisos | Recuperou exatamente `Baidu 百度`. Integração aprovada; amostra simples demais para medir precisão. |
| `assets/Unlimited-OCR.png` (925 × 283) | CLI de imagem | 218 s | 7.035 MiB no `nvidia-smi`; utilização amostrada de até 70% | JSON 1.0, 1 página, 1 bloco `image`, aviso `empty_block` | Retornou somente `<|det|>image [0, 0, 999, 999]<|/det|>`. Não recuperou texto útil do diagrama. |
| Página 1 do Form 1040 de 2025, 200 DPI (1700 × 2200) | CLI de imagem sobre página renderizada | 388 s | 7.039 MiB no `nvidia-smi`; utilização amostrada de até 98% | 13.143 caracteres, 16 linhas e 16 blocos: 6 `header`, 9 `table` e 1 `text` | Leu parte superior, entrou em repetição, não chegou a `adjusted gross income` e terminou truncado. Resultado reprovado para uso. |

O PDF usado foi o [Form 1040 oficial do IRS](https://www.irs.gov/pub/irs-pdf/f1040.pdf),
arquivo de 220.237 bytes com SHA-256
`3d31c226df0d189ced80e039d01cf0f8820c1019681a0f0ca6264de277b7e982`
na data do teste. Na saída da primeira página, `Check if you` apareceu 397
vezes; havia 9 aberturas de `<table>` e somente 8 fechamentos. Os 16 marcadores
de detecção estavam balanceados, e o JSON continuou sintaticamente válido.
Além disso, aquele JSON salvo não recebeu aviso do parser. O verificador do
fluxo de PDF detecta o desbalanceamento de tabelas separadamente, mas esse caso
mostra que ainda há espaço para ampliar os avisos do conversor.

### Como repetir o teste do Form 1040

O download abaixo busca uma fonte pública. Confira o hash porque o IRS pode
substituir o arquivo no mesmo endereço:

```bash
mkdir -p outputs/repro
curl -L https://www.irs.gov/pub/irs-pdf/f1040.pdf \
  -o outputs/repro/irs-f1040-2025.pdf
sha256sum outputs/repro/irs-f1040-2025.pdf

python - <<'PY'
from pathlib import Path
import fitz

source = Path("outputs/repro/irs-f1040-2025.pdf")
target = Path("outputs/repro/irs-f1040-page-1.png")
with fitz.open(source) as document:
    page = document.load_page(0)
    page.get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72), alpha=False).save(target)
print(target)
PY

python examples/transformers_image_to_json.py \
  outputs/repro/irs-f1040-page-1.png \
  --mode base --max-length 4096 \
  --revision 07dea832e22aefee32ad281d4b80551282e1c168 \
  --output outputs/repro/irs-f1040-page-1.json \
  --raw-output outputs/repro/irs-f1040-page-1.raw.txt
```

Para acompanhar a GPU em outro terminal:

```bash
nvidia-smi --query-gpu=timestamp,name,memory.used,utilization.gpu \
  --format=csv --loop-ms=500
```

Os tempos incluem a execução completa medida naquele ambiente e não são uma
garantia de desempenho. Cache do modelo, frequência/energia da GPU, programas
em segundo plano, tamanho visual e quantidade de tokens mudam o resultado.
Picos medidos por alocação do PyTorch e memória de processo do `nvidia-smi`
também não são métricas idênticas.

### Smoke test não é benchmark

- **Smoke test:** confirma que ambiente, pesos, CUDA, inferência, parser e
  arquivos de saída funcionam juntos. O logo cumpre essa função.
- **Teste de estresse:** explora uma entrada mais difícil para descobrir falhas.
  O diagrama e o formulário revelaram classificações vazias, repetição e
  truncamento.
- **Benchmark de precisão:** exige um conjunto representativo, transcrição de
  referência e métricas como CER/WER, além de avaliação própria para layout e
  tabelas. Esse benchmark ainda não foi realizado neste fork.

Portanto, `exit_code=0`, JSON válido ou GPU ocupada não significam que o texto
esteja correto. Sempre confronte o texto bruto e as caixas com a prévia visual.

## 5. PDF difícil, página por página

Este fluxo renderiza cada página a 300 DPI e mantém a concorrência fixa em 1.
Assim, nenhuma página depende do contexto da anterior e o pico de memória fica
mais previsível.

PowerShell:

```powershell
.\.venv\Scripts\python.exe examples\difficult_pdf_to_json.py C:\docs\contrato.pdf --mode base --max-length 4096 --output-dir outputs\contrato
```

WSL/Linux:

```bash
python examples/difficult_pdf_to_json.py /mnt/c/docs/contrato.pdf \
  --mode base --max-length 4096 --output-dir outputs/contrato
```

Os comandos acima usam a configuração conservadora validada na GPU de 8 GB.
Quando houver mais VRAM disponível, o modo `auto` começa com `base`. Se a
resposta estiver vazia, curta,
desbalanceada ou aparentemente repetitiva, tenta `gundam` e mantém o melhor
resultado disponível. Se até o modo `base` esgotar a VRAM, o script não tenta
`gundam`, que usa ainda mais recortes. Falhas e falta de memória são registradas
em `processing.pages`; o script continua nas páginas seguintes.

Arquivos gerados:

```text
outputs/contrato/
├── document.json
└── raw/
    ├── page_0001.txt
    ├── page_0002.txt
    └── ...
```

Use `--include-raw` para também duplicar as respostas originais dentro de
`document.json`. Use `--mode base` quando quiser impedir a tentativa de
`gundam`. PDFs protegidos podem receber `--password`, lembrando que argumentos
de linha de comando podem ficar no histórico do terminal.

## 6. Aprendizados, limitações e privacidade

- Os pesos BF16 ocupam aproximadamente 6–7 GB antes de imagens, cache e
  estruturas do PyTorch. Uma GPU de 8 GB fica no limite. Feche outros programas
  que usam a GPU, comece com `base` e mantenha `--max-length` abaixo do limite
  oficial de 32768. Reduzir demais o limite pode truncar páginas longas.
- `max_length` é o tamanho total da sequência, incluindo tokens da imagem e do
  prompt; não é apenas o número de tokens novos.
- `gundam` cria vários recortes em páginas grandes. Em 8 GB ele pode falhar
  mesmo quando `base` funciona; o exemplo captura esse OOM e segue em frente.
- `base` caber na memória não garante qualidade. Nos testes, ele acertou o logo,
  mas tratou o diagrama inteiro como imagem e repetiu conteúdo no formulário.
- A prévia de PDF usa renderização local por CPU e não carrega o modelo. Ela é
  útil para escolher página e DPI, mas não prevê se a geração do OCR será boa.
- Aumentar o DPI cria mais pixels e pode ajudar letras pequenas, porém também
  aumenta tempo, RAM e risco de atingir o limite de 25 milhões de pixels. Não
  recupera informação que já foi perdida por desfoque ou compressão.
- A primeira execução baixa pesos e código do Hugging Face. Como o carregamento
  usa `trust_remote_code=True`, revise o código e use `--revision <commit>` para
  fixar uma versão em ambientes controlados.
- Os logs dos testes registraram avisos do caminho Transformers sobre
  `torch_dtype`, `position_ids`, `attention_mask` e `pad_token_id`. As execuções
  terminaram com código zero, mas esses avisos devem ser preservados ao
  investigar comportamento ou comparar versões.
- Depois do download, os scripts processam os arquivos localmente. Não envie
  documentos confidenciais para demos públicas. O JSON contém o caminho local
  da origem e, no exemplo de imagem, também contém a resposta bruta.
- A interface evita o caminho absoluto no campo `source`, mas grava texto bruto
  e JSON em `outputs/web-ui/`. Esses arquivos continuam no disco depois de
  liberar a GPU ou fechar a página; apague-os conscientemente quando não forem
  mais necessários.
- `outputs/`, `tmp/`, `.env*` (salvo o nome de template `.env.example`),
  credenciais, chaves, logs, pesos e caches estão no `.gitignore`, mas antes de
  publicar use `git status`, `git diff --cached` e uma varredura de segredos.
  `.gitignore` não remove um segredo já commitado.
- 300 DPI ajuda com texto pequeno, mas não corrige sozinho desfoque, páginas
  tortas, sombras ou escrita manual. Faça pré-processamento e valide nomes,
  valores, tabelas e fórmulas antes de usar o resultado em produção.
- O conversor JSON é determinístico: ele preserva o que o modelo retornou. JSON
  válido comprova estrutura serializável, não veracidade do OCR. Avisos e
  heurísticas reduzem risco, mas não substituem revisão humana.
- Os 24 testes offline validam o parser, os controles do fluxo e a prévia com
  respostas salvas, exemplos públicos, fixtures e mocks. Eles não executam o
  modelo, não medem a precisão do OCR e não comprovam desempenho em PDFs
  difíceis.

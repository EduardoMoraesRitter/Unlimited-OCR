# Exemplos: OCR para JSON

Estes exemplos preservam a resposta original do Unlimited-OCR e a convertem
deterministicamente para JSON. Pedir "responda em JSON" ao modelo não é
necessário e pode produzir JSON inválido; a conversão é feita por
`unlimited_ocr_json.py` depois da inferência.

Execute os comandos a partir da raiz do repositório.

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
py -3.12 -m unittest discover -s tests -v
```

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
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python examples/transformers_image_to_json.py /mnt/c/docs/pagina.png --mode base --max-length 8192
```

A verificação precisa terminar com `True`. Se mostrar `False`, não prossiga com
o download dos pesos: o Python está usando uma edição sem CUDA. O Windows nativo
continua adequado para o conversor JSON e para os 18 testes offline, que não
carregam o modelo.

O script chama `model.infer(..., eval_mode=True)`. Por padrão, grava
`outputs/pagina.raw.txt` e `outputs/pagina.json`. Use `--output` e
`--raw-output` para escolher outros destinos.

Modos disponíveis:

- `base`: uma visão global de 1024 × 1024; consome menos memória.
- `gundam`: visão global e recortes de 640 × 640; pode recuperar letras
  pequenas, mas usa mais VRAM e demora mais.

## 3. Interface web local, estilo Hugging Face

Depois de concluir a instalação CUDA da seção anterior, instale a combinação
testada da interface sem alterar as versões do Transformers e do Pillow:

```bash
python -m pip install gradio==6.15.1 accelerate==1.14.0 huggingface-hub==0.36.0
python examples/gradio_cuda_app.py
```

Abra `http://127.0.0.1:7860`. A tela aceita PNG, JPEG, WebP, BMP ou uma página
selecionada de PDF, mostra a GPU CUDA detectada, preserva o texto bruto, exibe o
JSON e oferece o arquivo `.json` para download. Há dois exemplos rápidos já
incluídos no repositório.

O modelo só é carregado no primeiro clique em **Executar OCR na GPU**. As
requisições são serializadas e usam apenas o modo `base`, adequado ao limite de
8 GB da RTX 4070 testada. A interface limita a sequência a 4.096, o upload a 25
MB e a imagem renderizada a 25 milhões de pixels. O botão **Liberar memória da
GPU** remove os pesos sem fechar a página. O servidor usa apenas `127.0.0.1` e
`share=False`: ele não cria uma URL pública nem envia o documento para um
Gradio Space.

No teste local de 2026-08-06, `assets/baidu.png` retornou `Baidu 百度` em 92
segundos, gerou JSON válido e atingiu pico de 6.835 MiB de alocação CUDA pelo
PyTorch. Esse é um smoke test de integração, não uma medição de precisão.

## 4. PDF difícil, página por página

Este fluxo renderiza cada página a 300 DPI e mantém a concorrência fixa em 1.
Assim, nenhuma página depende do contexto da anterior e o pico de memória fica
mais previsível.

PowerShell:

```powershell
.\.venv\Scripts\python.exe examples\difficult_pdf_to_json.py C:\docs\contrato.pdf --mode auto --max-length 8192 --output-dir outputs\contrato
```

WSL/Linux:

```bash
python examples/difficult_pdf_to_json.py /mnt/c/docs/contrato.pdf --mode auto --max-length 8192 --output-dir outputs/contrato
```

O modo `auto` começa com `base`. Se a resposta estiver vazia, curta,
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

## Memória, qualidade e privacidade

- Os pesos BF16 ocupam aproximadamente 6–7 GB antes de imagens, cache e
  estruturas do PyTorch. Uma GPU de 8 GB fica no limite. Feche outros programas
  que usam a GPU, comece com `base` e mantenha `--max-length` abaixo do limite
  oficial de 32768. Reduzir demais o limite pode truncar páginas longas.
- `max_length` é o tamanho total da sequência, incluindo tokens da imagem e do
  prompt; não é apenas o número de tokens novos.
- `gundam` cria vários recortes em páginas grandes. Em 8 GB ele pode falhar
  mesmo quando `base` funciona; o exemplo captura esse OOM e segue em frente.
- A primeira execução baixa pesos e código do Hugging Face. Como o carregamento
  usa `trust_remote_code=True`, revise o código e use `--revision <commit>` para
  fixar uma versão em ambientes controlados.
- Depois do download, os scripts processam os arquivos localmente. Não envie
  documentos confidenciais para demos públicas. O JSON contém o caminho local
  da origem e, no exemplo de imagem, também contém a resposta bruta.
- 300 DPI ajuda com texto pequeno, mas não corrige sozinho desfoque, páginas
  tortas, sombras ou escrita manual. Faça pré-processamento e valide nomes,
  valores, tabelas e fórmulas antes de usar o resultado em produção.
- Os 18 testes offline validam o parser e os controles do fluxo com respostas
  salvas, fixtures e mocks. Eles não executam o modelo, não medem a precisão do
  OCR e não comprovam desempenho em PDFs difíceis.

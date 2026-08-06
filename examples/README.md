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

Instale primeiro as dependências da seção Transformers do
[`README.md`](../README.md). O ambiente validado pelo projeto usa Python 3.12,
CUDA, PyTorch 2.10.0, Transformers 4.57.1 e Pillow 12.1.1.

PowerShell no Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 transformers==4.57.1 Pillow==12.1.1 matplotlib==3.10.8 einops==0.8.2 addict==2.4.0 easydict==1.13 pymupdf==1.27.2.2 psutil==7.2.2
.\.venv\Scripts\python.exe examples\transformers_image_to_json.py C:\docs\pagina.png --mode base --max-length 8192
```

WSL/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 transformers==4.57.1 Pillow==12.1.1 matplotlib==3.10.8 einops==0.8.2 addict==2.4.0 easydict==1.13 pymupdf==1.27.2.2 psutil==7.2.2
python examples/transformers_image_to_json.py /mnt/c/docs/pagina.png --mode base --max-length 8192
```

O script chama `model.infer(..., eval_mode=True)`. Por padrão, grava
`outputs/pagina.raw.txt` e `outputs/pagina.json`. Use `--output` e
`--raw-output` para escolher outros destinos.

Modos disponíveis:

- `base`: uma visão global de 1024 × 1024; consome menos memória.
- `gundam`: visão global e recortes de 640 × 640; pode recuperar letras
  pequenas, mas usa mais VRAM e demora mais.

## 3. PDF difícil, página por página

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

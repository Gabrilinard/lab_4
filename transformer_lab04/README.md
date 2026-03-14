---

# Instalação do Projeto

## 1. Clonar o Repositório

Primeiro, clone o repositório para sua máquina:

```bash
git clone https://github.com/Gabrilinard/lab_4
```

Depois entre na pasta do projeto:

```bash
cd lab_4/transformer_lab04/transformer
```

---

# Estrutura do Projeto

A estrutura principal do projeto é:

```
lab_4
 └ transformer_lab04
     └ transformer
        ├ tarefa01.py
        ├ tarefa02.py
        ├ tarefa03.py
        ├ tarefa04.py
        ├ embedding.py
        ├ mask.py
        └ README.md
```

Cada arquivo representa uma parte da implementação do Transformer desenvolvida no laboratório.

---

# Requisitos

Para executar o projeto é necessário ter instalado:

- **Python 3.9 ou superior**
- **PyTorch**
- **Git**

---

# Instalação das Dependências

Instale o PyTorch executando:

```bash
pip install torch
```'

---

# Verificar Instalação

Para verificar se o PyTorch foi instalado corretamente:

```python
import torch
print(torch.__version__)
```

Se aparecer a versão instalada, o ambiente está pronto para executar o projeto.

---

# Laboratório Transformer: Implementação de Inferência Auto-Regressiva

Este repositório apresenta a implementação completa de um **modelo Transformer**, construída de forma modular ao longo de quatro tarefas principais.  
O objetivo do laboratório é demonstrar o **fluxo completo de dados da arquitetura Transformer**, desde o cálculo de atenção básica até a **geração auto-regressiva de tokens durante a inferência**.

A implementação utiliza **PyTorch**

---

# Documentação da Lógica Matemática

A implementação foi fundamentada em conceitos de **álgebra linear**, **probabilidade** e **Processamento de Linguagem Natural (NLP)** aplicados à arquitetura Transformer.

---

# 1. Mecanismo de Atenção (Tarefa 01)

O núcleo da arquitetura Transformer é o mecanismo de **Scaled Dot-Product Attention**.

A atenção mede a similaridade entre **Query (Q)** e **Key (K)**, gerando pesos que serão aplicados sobre **Value (V)**.

A fórmula é:

\[
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

### Explicação da fórmula

- **Q (Query)** ->  vetor que representa o elemento que busca informação.
- **K (Key)** -> vetor que representa o conteúdo disponível.
- **V (Value)** -> vetor que contém a informação a ser combinada.
- **d_k** -> dimensão das keys.

A divisão por:

\[
\sqrt{d_k}
\]

serve para **evitar valores muito grandes antes do softmax**, mantendo a estabilidade dos gradientes durante o treinamento.

---

### Implementação da máscara

No **Decoder**, o modelo não pode acessar palavras futuras durante o treinamento ou inferência.

Por isso aplicamos uma **máscara causal**, que transforma posições inválidas em **−∞**, fazendo o softmax gerar probabilidade zero.

```python
if mascara is not None:
    scores = scores.masked_fill(mascara == 0, float("-inf"))
```

---

# 2. Multi-Head Attention

A **Multi-Head Attention (MHA)** permite que o modelo observe diferentes relações semânticas ao mesmo tempo.

O vetor de entrada é dividido em várias "cabeças" de atenção.

Fluxo:

```
Input -> Linear projections (Q,K,V) -> Split Heads -> Scaled Dot-Product Attention -> Concat Heads -> Linear Projection
```

Cada cabeça aprende **padrões distintos de relacionamento entre palavras**.

---

# 3. Feed Forward Network (FFN)

Após a atenção, cada token passa por uma **rede neural totalmente conectada**.

\[
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
\]

Características:

- aplicada **independentemente a cada posição**
- aumenta a capacidade de representação
- usa **ReLU** como ativação

---

# 4. Conexões Residuais (Add & Norm)

Cada subcamada do Transformer possui:

```
Add + LayerNorm
```

Formalmente:

\[
LayerNorm(x + Sublayer(x))
\]

Isso ajuda a:

- evitar **vanishing gradients**
- estabilizar treinamento
- permitir redes mais profundas

---

# 5. Encoder Stack (Tarefa 02)

O **Encoder** transforma a sequência de entrada em uma representação contextual chamada **memória rica (Z)**.

Cada camada do Encoder contém:

```
Self Attention -> Add & Norm -> Feed Forward -> Add & Norm
```

Fluxo geral:

```
Input Embedding -> Positional Encoding -> Encoder Layer 1 -> Encoder Layer 2 -> ... Encoder Layer N -> Memória Z
```

A saída do Encoder é uma matriz:

```
Z ∈ R^(batch × seq_len × d_model)
```

Essa matriz contém **informação contextual de toda a frase**.

---

# 6. Decoder Stack (Tarefa 03)

O **Decoder** é responsável por gerar a sequência de saída.

Cada camada do Decoder possui **três subcamadas**:

```
Masked Self-Attention -> Add & Norm -> Cross Attention (Encoder-Decoder) -> Add & Norm -> Feed Forward -> Add & Norm
```

### Cross-Attention

Aqui ocorre a conexão entre Encoder e Decoder:

```
Q -> sequência gerada pelo decoder
K -> memória do encoder
V -> memória do encoder
```

Isso permite que o modelo **focalize nas partes relevantes da frase de entrada**.

---

# 7. Positional Encoding

Como Transformers não possuem recorrência, precisamos informar **a posição das palavras**.

Foi utilizado **Positional Encoding senoidal**, definido por:

\[
PE(pos,2i) = sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos,2i+1) = cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

Isso permite que o modelo capture **relações de ordem entre tokens**.

---

# 8. Máscara Causal

A máscara causal impede que o Decoder acesse tokens futuros.

Exemplo para sequência de tamanho 4:

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Implementação:

```python
mask = torch.tril(torch.ones(seq_len, seq_len))
```

---

# 9. Inferência Auto-Regressiva (Tarefa 04)

A geração de texto no Transformer é **auto-regressiva**.

Isso significa que cada token gerado é usado como entrada para prever o próximo.

Fluxo:

```
<START>
Decoder -> Next token -> Append -> Decoder novamente
```

Loop principal:

```python
while len(lista_indices_gerados) < max_len:
    probabilidades = model.decode(decoder_input_indices, z, tgt_mask)
    next_idx = torch.argmax(probabilidades[0, -1, :]).item()
```

Passos da inferência:

1. Encoder processa a frase de entrada
2. Decoder inicia com `<START>`
3. Modelo prevê próximo token
4. Token é anexado à sequência
5. Processo repete até `<EOS>` ou limite máximo

---

# Estrutura de Arquivos

```
tarefa01.py
Implementação do mecanismo de Atenção, Multi-Head Attention, FFN e Add & Norm

tarefa02.py
Construção da pilha do Encoder

tarefa03.py
Construção da pilha do Decoder e projeção para vocabulário

tarefa04.py
Script final integrando Encoder, Decoder e inferência auto-regressiva

embedding.py
Camada de Embedding + Positional Encoding

mask.py
Geração de máscaras causais e máscaras de padding
```

---

# Como Executar

Para executar a simulação completa da arquitetura Transformer:

```bash
python tarefa04.py
```

O script executará uma inferência simulada para a frase:

```
Thinking Machines
```

Saída esperada (exemplo):

```
Entrada (IDs): [[10, 11]] -> 'Thinking Machines'
Iniciando loop auto-regressivo...

Passo 1: Previu ID 348
Passo 2: Previu ID 125
Passo 3: Previu ID 2

Token <EOS> detectado.
Resultado Final: [1, 348, 125, 2]
```

Como o modelo **não foi treinado**, os tokens gerados são aleatórios.

---

# Nota de Integridade Acadêmica e Uso de Inteligência Artificial

Este projeto foi desenvolvido por **Gabriel Linard Leite** como parte dos requisitos do **Laboratório Técnico 04** do **iCEV**.

Em conformidade com o **Contrato Pedagógico da disciplina**, declaro que o trabalho apresentado foi desenvolvido com participação ativa do autor e que qualquer apoio de ferramentas externas foi utilizado apenas como **suporte técnico**, não substituindo a compreensão ou implementação do conteúdo.

---

# Uso de Inteligência Artificial

Durante o desenvolvimento deste projeto, ferramentas de **Inteligência Artificial** foram utilizadas como **apoio auxiliar** em algumas etapas específicas do processo. Todo o conteúdo gerado foi **integralmente revisado, adaptado e validado manualmente** pelo autor (Gabriel Linard Leite).

---

# Atribuições do Uso de IA

A utilização de IA ocorreu nas seguintes atividades:

### 1. Estruturação e Templates de Código
A IA foi utilizada para auxiliar na geração inicial de **estruturas base de classes (boilerplate)** e na organização geral da arquitetura do projeto.

Essas sugestões serviram como ponto de partida para a implementação dos módulos principais do Transformer.

---

### 2. Organização e Formatação da Documentação

Ferramentas de IA foram utilizadas para auxiliar na:

- organização da documentação técnica
- estruturação do arquivo **README.md**
- melhoria da clareza na explicação dos conceitos matemáticos da arquitetura Transformer

Todo o conteúdo foi posteriormente **editado e validado manualmente**.

---

### 3. Refatoração e Adaptação de Código

A IA auxiliou no processo de **refatoração de funções matemáticas originalmente implementadas em NumPy para PyTorch**, garantindo:

- compatibilidade com tensores do PyTorch
- manutenção das dimensões corretas
- consistência com os requisitos técnicos da **Tarefa 01**

---

### 4. Revisão Técnica do Fluxo da Arquitetura

A ferramenta também foi utilizada para **verificação conceitual** do fluxo de dados da arquitetura Transformer, especialmente nos seguintes pontos:

- propagação de tensores entre Encoder e Decoder
- funcionamento da **Cross-Attention**
- validação da coerência com a implementação teórica de um **Transformer implementado "From Scratch"**

Essa etapa serviu apenas como **checagem conceitual**, sendo toda a implementação analisada e testada manualmente por mim, Gabriel Linard Leite.

---

# Declaração de Responsabilidade

Declaro que:

- compreendo plenamente toda a lógica implementada no projeto;
- todas as decisões de implementação foram **avaliadas e aprovadas por mim**;
- o código final reflete **meu entendimento da arquitetura Transformer**.

A utilização de Inteligência Artificial foi restrita a **suporte técnico e organizacional**, não substituindo o processo de aprendizado nem a autoria do trabalho.

---* foram revisadas, implementadas e testadas manualmente por mim para garantir o cumprimento dos requisitos pedagógicos da atividade.

---

# Conclusão

Este laboratório demonstra na prática os principais componentes da arquitetura Transformer:

- Scaled Dot-Product Attention
- Multi-Head Attention
- Encoder Stack
- Decoder Stack
- Cross Attention
- Positional Encoding
- Inferência Auto-Regressiva

A implementação modular permite compreender **como cada componente contribui para o funcionamento completo do modelo**.

---

# Próximos Passos

Possíveis melhorias futuras:

- Implementar **treinamento do modelo**
- Adicionar **loss function (CrossEntropy)**
- Implementar **teacher forcing**
- Criar **dataset real de linguagem**
- Avaliar métricas de geração de texto

---
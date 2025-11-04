# RedeNeural (Perceptron com sigmoide)

Projeto de exemplo que implementa uma rede neural de camada única (perceptron) usando NumPy e matplotlib.

Descrição
- Rede de uma camada com função de ativação sigmoide.
- Treinamento via retropropagação simples (gradiente usando derivada da sigmoide) e MSE como métrica.
- Código principal em [main.py](main.py).

Principais funções (no arquivo [main.py](main.py))
- [`main.sigmoide`](main.py): função de ativação sigmoide.
- [`main.derivada_sigmoide`](main.py): derivada da sigmoide usada no cálculo de ajuste.
- [`main.inicializar_pesos`](main.py): inicializa os pesos sinápticos.
- [`main.treinar_rede`](main.py): loop de treinamento que retorna pesos finais e histórico de erro.
- [`main.avaliar_rede`](main.py): computa a saída da rede com pesos treinados.
- Dados de exemplo: [`main.entradas`](main.py) e [`main.saidas`](main.py).

Requisitos
- Python 3.8+
- numpy
- matplotlib

Instalação
```sh
pip install numpy matplotlib
```

Como executar
```sh
python main.py
```
O script:
- Treina a rede por `num_interacoes` (configurado em `main.py`).
- Imprime o erro quadrático médio a cada 1000 iterações.
- Mostra os pesos finais e as saídas calculadas.
- Plota a curva de erro ao longo das iterações.

O que observar
- Ajuste `num_interacoes` e `taxa_aprendizado` em [main.py](main.py) para obter convergência mais rápida ou estável.
- A semente (`np.random.seed(1)`) em [`main.inicializar_pesos`](main.py) garante reprodutibilidade.

Estrutura do repositório
- main.py — implementação da rede e script de execução. ([main.py](main.py))
- import numpy as np.txt — arquivo sobrando no repositório que pode ser removido. ([import numpy as np.txt](import numpy as np.txt))

Licença
- MIT
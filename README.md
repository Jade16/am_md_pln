# Reconhecimento de Entidade Nomeada em Textos de Época

## Arquivos e partes do projeto
### Modelos e datasets
O projeto se utiliza de três datasets e modelos associados, contidos no diretório datasets/.
- Um dataset em inglês de discursos das nações unidas, que é utilizado como base para comparação com outros estudos e intodução incial da metodologia. É utilizdo o modelo base do BERT destilado.
- Um dataset em inglês com os textos de época traduzidos utilizando o serviço gratuito do google tradutor, utilizado para analisar a capacidade de generalização do BERT destilado usado no caso anterior
- Um dataset em português com os textos de época como proposto no projeto inicial.

Os códigos de configuração e extração de dados estão disponíveis nos arquivos un.py, en.py e pt.py, respectivamente, e identificados por esses nomes em todo o código.

### Exploração dos dados

Além disso, data_exploration.py contém uma análise exploratória uniforme para todos os modelos utilizando streamlit, com histogramas, gráficos em barra e uma wordcloud, úteis para a tarefa de NER.

### Treinamento e avaliação
trainer.py contém o código necessário para treinar e avaliar os modelos de forma genérica, colocando os resultados da avaliação como gráficos e valores no streamlit.

### Configuração e execução
O arquivo config.py contém as flags de configuração com parâmetros para escolha do modelo e dataset, hiperparâmetros de treinamento e escolha do dispositivo a ser utilizado (cpu ou gpu).

Por fim, o arquivo main.py controla a execução do código completo.

## Executando o código
Inicialmente, instale as bibliotecas utilizadas no projeto. Para usuários de NixOS, basta utilizar `nix develop`.

Após isso, execute `streamlit run src/main.py -- <argumentos>`, para ver os possíveis argumentos, use `streamlit run src/main.py -- --help`

## Feito totalmente por
- Lucas Eduardo Gulka Pulcinelli
- Jade Bortot de Paiva 

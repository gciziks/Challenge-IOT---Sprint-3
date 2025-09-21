# Sistema de Reconhecimento Facial

Este projeto implementa um sistema de detecção e reconhecimento facial em tempo real utilizando OpenCV e o algoritmo LBPH (Local Binary Patterns Histograms).  
Ele permite coletar dados faciais de novos usuários, treinar o modelo e reconhecer rostos pela câmera em tempo real, com parâmetros ajustáveis diretamente pelo teclado.

---

## Objetivo
- Detectar rostos em imagens capturadas pela webcam.
- Reconhecer usuários cadastrados com base em dados previamente coletados.
- Permitir ajustes de parâmetros de detecção e reconhecimento em tempo real.
- Possibilidade de adicionar novos usuários e retreinar o modelo.

---

## Dependências

Antes de rodar, instale as bibliotecas necessárias:

```bash
pip install opencv-contrib-python numpy
```

> É importante instalar **opencv-contrib-python**, pois o reconhecedor LBPH (`cv2.face.LBPHFaceRecognizer_create`) só está disponível nessa versão.

---

## Execução

1. Clone ou baixe este repositório.
2. Garanta que o arquivo principal (`face_recognizer.py`) e o script de execução estejam na mesma pasta.
3. Execute o programa:

```bash
python main.py
```

4. Na primeira execução, se não houver modelo treinado, o sistema perguntará se deseja treinar um novo usuário.
5. Para coletar dados:
   - Informe um nome de usuário.
   - Fique em frente à câmera.
   - Movimente levemente a cabeça para capturar diferentes ângulos.
   - Serão coletadas múltiplas imagens (mínimo de 10).
   - Ao final, o sistema treina automaticamente.

---

## Controles

Durante a execução do reconhecimento:

- **1-4** → Ajustar Scale Factor (1.05 - 1.5)  
- **5-8** → Ajustar Min Neighbors (1 - 10)  
- **9** → Resetar parâmetros para padrão  
- **T** → Treinar novo usuário  
- **R** → Retreinar o modelo  
- **H** → Mostrar/Ocultar ajuda na tela  
- **ESC** → Sair do sistema  

---

## Parâmetros Importantes

- **Scale Factor**: controla a sensibilidade da detecção (valores maiores = menos detecções).  
- **Min Neighbors**: número de vizinhos para confirmar uma face (valores maiores = menos falsos positivos).  
- **Min Size**: tamanho mínimo da face a ser detectada (padrão: 30x30).  
- **Confidence Threshold**: limiar de confiança para considerar uma face reconhecida (padrão: 70).  

Esses valores podem ser ajustados dinamicamente pelo teclado durante a execução.

---

## Estrutura de Arquivos

```
facial_dataset/         # Pasta criada automaticamente para armazenar imagens de usuários
face_recognizer.yml     # Arquivo com o modelo treinado (gerado após treinamento)
labels.pickle           # Mapeamento dos rótulos (id → nome do usuário)
face_recognizer.py      # Classe principal com toda a lógica do sistema
main.py                 # Script de inicialização
```

---

## Melhorias Futuras
- Implementar um banco de dados para armazenar informações dos usuários.

---
## Nota Ética sobre Uso de Dados Faciais
O uso de tecnologias de reconhecimento facial levanta questões importantes sobre privacidade, consentimento e segurança dos dados.  
Este projeto tem finalidade educacional e não deve ser utilizado em ambientes de produção sem:
- Consentimento explícito dos usuários cujos dados faciais forem coletados.
- Mecanismos de segurança para proteger os dados armazenados.
- Transparência sobre como os dados são usados e por quanto tempo são mantidos.

Recomenda-se seguir sempre as leis de proteção de dados pessoais vigentes como a LGPD


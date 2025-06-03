#  Animal Classifier API – Reconhecimento de Animais com IA

## 📌 Funcionalidades

- Classificação de imagens em **10 classes de animais** (cachorro, gato, cavalo, elefante, etc.).
- API REST robusta construída com **FastAPI**.
- Interface web amigável via **Streamlit**.
- **Armazenamento automático** das imagens recebidas, útil para reuso ou aprendizado contínuo.
- Arquitetura modular e extensível, pronta para melhorias.

---

## ⚠️ Observações Importantes

- O modelo treinado (`cnn_animals10.h5`) **não está incluído neste repositório** devido ao seu tamanho elevado.
- A base de dados Animals-10 **também não está incluída**, pois ultrapassa os limites de armazenamento do GitHub.
- Recomendo baixar os dados manualmente e treinar o modelo localmente seguindo as instruções abaixo.

---

## 📈 Resultado Atual

- **Acurácia de teste:** ~70%
- Desempenho ideal com imagens bem iluminadas e com características claras.
- Em desenvolvimento: aprimoramentos na arquitetura da CNN e aumento de dados para mais robustez.

---

## 🧰 Tecnologias Utilizadas

- Python
- TensorFlow / Keras – para o modelo de rede neural
- FastAPI – para a API REST
- Streamlit – para interface web
- Pillow & NumPy – para processamento de imagens
- Animals-10 Dataset – base com 10.000+ imagens  
  🔗 [Link para download no Kaggle](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

---

## 📦 Pré-requisitos

- Python 3.8 ou superior
- Instalação das dependências:
  ```bash
  pip install -r requirements.txt
  ```

---

## ▶️ Passo a Passo para Rodar o Projeto

1. **Baixe e organize a base de dados:**
   - Acesse o Animals-10 Dataset no Kaggle.
   - Extraia os diretórios para `data/raw/`.
   - Os diretórios devem conter os nomes originais em italiano (como `cane`, `gatto`, etc.).

2. **Divida a base em treino, validação e teste:**
   ```bash
   python script/dados_divididos.py
   ```
   - Isso criará as pastas `data/split/train`, `val` e `test`.

3. **Pré-processamento dos dados:**
   ```bash
   python script/preprocess.py
   ```
   - Organiza e prepara os dados para treino dentro de `data/preprocessed/`

4. **Treine o modelo de classificação:**
   ```bash
   python training/train.py
   ```
   - Gera o modelo treinado (`cnn_animals10.h5`) e salva na pasta `model/`.

5. **Inicie a API com FastAPI:**
   ```bash
   uvicorn api.main:app --reload
   ```
   - Acesse no navegador: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

6. **(Opcional) Inicie a interface web com Streamlit:**
   ```bash
   streamlit run frontend/frontend.py
   ```

---

## 💡 Melhorias Futuras

- Melhorar a arquitetura da CNN com mais camadas e regularização.
- Aplicar técnicas de aumento de dados com `ImageDataGenerator`.

---

## 📂 Estrutura Esperada de Diretórios

```
├── api/
│   ├── uploads/
│   └── main.py
│
├── data/
│   ├── preprocessed/
│   │   ├── test_ds/
│   │   ├── train_ds/
│   │   └── val_ds/
│   │
│   ├── raw/
│   │   ├── cane/       
│   │   ├── cavallo/     
│   │   ├── etc...
│   │   
│   └── split/
│       ├── test/
│       ├── train/
│       └── val/
│
├── frontend/
│   └── frontend.py
│
├── model/
│   └── cnn_animals10.h5
│
├── script/
│   ├── dados_divididos.py
│   └── preprocess.py
│
├── training/
│   └── train.py
│
├── uploads/
├── .gitignore
└── README.md
```
---


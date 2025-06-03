import splitfolders #ferramenta para dividir automaticamente conjuntos de dados (em pasta) em treino, validação e teste

splitfolders.ratio(
    "data/raw",                
    output="data/split",     
    seed=42,
    ratio=(.8, .1, .1),        # 80% treino, 10% val, 10% teste
    move=True                  # ou False, se quiser manter os originais
)


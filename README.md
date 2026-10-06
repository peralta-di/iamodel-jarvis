# Jarvis: mi propio modelo de IA desde cero

Jarvis es un modelo de lenguaje tipo GPT construido **desde cero** con Python y PyTorch: el tokenizador, la atención, el transformer, el entrenamiento y la generación de texto.

El proyecto sigue la sección [Build your own AI Model](https://github.com/codecrafters-io/build-your-own-x#build-your-own-ai-model) de *build-your-own-x*, y en concreto el tutorial [**LLMs-from-scratch**](https://github.com/rasbt/LLMs-from-scratch) de Sebastian Raschka. Cada archivo corresponde a un capítulo de ese tutorial:

| Capítulo del tutorial | Archivo | Qué hace |
|---|---|---|
| 2. Trabajar con texto | `jarvis/tokenizer.py` | Tokenizador BPE propio, entrenado con tus textos (funciona bien en español) |
| 2. Trabajar con texto | `jarvis/data.py` | Ventana deslizante: entrada → siguiente token |
| 3. Mecanismo de atención | `jarvis/model.py` | `MultiHeadAttention` con máscara causal |
| 4. Implementar GPT | `jarvis/model.py` | `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel` |
| 5. Preentrenamiento | `train.py` | Bucle de entrenamiento con pérdida de entrenamiento y validación |
| 5. Generar texto | `jarvis/generate.py`, `chat.py` | Muestreo con `temperature` y `top_k` |

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rápido

**1. Entrenar con el texto de ejemplo** (`data/jarvis_ejemplo.txt`, tarda menos de un minuto en CPU):

```bash
python train.py --vocab-size 512 --context-length 64 --batch-size 8 --epochs 30 --eval-every 50
```

El texto de ejemplo es muy corto, así que Jarvis escribirá un español todavía bastante raro. Sirve para comprobar que todo funciona.

**2. Hablar con Jarvis:**

```bash
python chat.py                              # modo interactivo
python chat.py --prompt "Jarvis, ¿qué es"   # una sola respuesta
python chat.py --temperature 0.3            # respuestas más conservadoras
```

## Entrenar un Jarvis mejor

Un modelo de lenguaje aprende de lo que lee: **cuanto más texto, mejor**.

1. Descarga un libro en español de dominio público (Don Quijote, unos 2 MB):

   ```bash
   python scripts/download_data.py
   ```

   O pon tus propios archivos `.txt` (apuntes, cuentos, conversaciones…) dentro de `data/`. El entrenamiento usa todos los `.txt` de esa carpeta.

2. Entrena con los valores por defecto (unos 4 M de parámetros):

   ```bash
   python train.py --epochs 3
   ```

   En CPU puede tardar una o varias horas; con una GPU (NVIDIA o Apple Silicon) es mucho más rápido y se usa automáticamente. Para una prueba corta añade `--max-steps 500`.

3. Si tienes GPU, prueba un modelo más grande:

   ```bash
   python train.py --emb-dim 384 --layers 6 --heads 6 --context-length 256 --vocab-size 4096
   ```

### Cómo leer la salida del entrenamiento

- **pérdida train / val**: cuánto se equivoca el modelo. Debe bajar.
- **perplejidad**: aproximadamente, entre cuántos tokens "duda" el modelo. Cuanto más baja, mejor.
- Si la pérdida de *train* baja pero la de *val* sube, el modelo está **memorizando** (sobreajuste): añade más texto, sube `--dropout` o entrena menos épocas. El modelo se guarda automáticamente en el punto con mejor pérdida de validación.

## Opciones de `train.py`

| Opción | Por defecto | Descripción |
|---|---|---|
| `--data` | `data` | Archivo `.txt` o carpeta con archivos `.txt` |
| `--out` | `checkpoints/jarvis` | Dónde se guarda el modelo |
| `--vocab-size` | 2048 | Tamaño del vocabulario del tokenizador |
| `--context-length` | 128 | Tokens que el modelo ve a la vez |
| `--emb-dim` / `--layers` / `--heads` | 256 / 4 / 4 | Tamaño del modelo (`emb-dim` debe ser divisible por `heads`) |
| `--epochs` / `--max-steps` | 10 / — | Duración del entrenamiento |
| `--lr` | 4e-4 | Tasa de aprendizaje |

## Pruebas

```bash
python -m pytest
```

## Próximos pasos

Las otras dos guías de la misma sección de *build-your-own-x* son buenas continuaciones:

- **Ajuste fino para conversar** (capítulo 7 de LLMs-from-scratch): entrenar a Jarvis con pares de pregunta/respuesta para que actúe como asistente.
- **[RAG from scratch](https://github.com/langchain-ai/rag-from-scratch)**: dejar que Jarvis busque en tus documentos antes de responder.
- **[Modelos de difusión](https://huggingface.co/learn/diffusion-course/en/unit1/3)**: generar imágenes en vez de texto.

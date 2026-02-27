# Informe 1-DGSI

## Investigación de conceptos básicos de IA
### Parte 1 --- Conceptos básicos para la investigación

#### Model weights

a.  *Definición*

> Se trata de parámetros de redes neuronales que se pueden aprender y
> que se ajustan durante el proceso de entrenamiento.

b.  *Por qué es importante en el aprendizaje automático*

> Esto es importante porque sirve como repositorio de toda la
> información que ingresa constantemente, lo que permite que el modelo
> reconozca patrones y haga predicciones precisas con nuevos datos.

c.  *Relación con GPT-2 o Transformadores*

> En GPT-2, los pesos del modelo contienen toda la información sobre el
> lenguaje, como la gramática, el contexto y las relaciones entre
> palabras, que ya está almacenada y se utiliza para ajustar el modelo
> constantemente. Cuando se carga un modelo preentrenado en la
> biblioteca Transformers, se carga el peso entrenado.

d.  *Ejemplo práctico*

> "from_pretained("gpt2")" descarga y carga los pesos preentrenados en
> la arquitectura GPT2. Estos pesos contienen el conocimiento adquirido
> durante el entrenamiento, lo que permite al modelo generar texto
> coherente sin necesidad de entrenamiento adicional.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar modelo y tokenizer

model = GPT2LMHeadModel.from_pretrained(\"gpt2\")

tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")

# Texto de entrada

input_text = \"Maquina de aprendizaje en ChatGPT\"

inputs = tokenizer.encode(input_text, return_tensors=\"pt\")

# Generar texto

outputs = model.generate(inputs, max_length=20)

print(tokenizer.decode(outputs\[0\]))

```

##### Neural Network

a.  *Definición*

> Es un modelo computacional que comienza con datos de entrada, los
> cuales mediante procesos matemáticos y ponderación producen una salida
> que puede dar lugar a la salida final o a la entrada a otra nueva red
> neuronal.

b.  *Por qué es importante en el aprendizaje automático*

> Es importante porque permite el aprendizaje de patrones complejos como
> el procesamiento del habla, el reconocimiento de texto y la traducción
> automática, gracias a su capacidad de aprender de grandes volúmenes de
> información.

c.  *Relación con GPT-2 o Transformadores*

> GPT-2 es esencialmente una red neuronal profunda basada en la
> arquitectura Transformer; además, la biblioteca Transformers ya tiene
> una red neuronal implementada que está especializada para cada modelo.

d.  *Ejemplo práctico*

> En este caso, gpt-2, que es una red neuronal especializada, recibe un
> texto de entrada que luego es procesado por sus capas utilizando
> patrones aprendidos para generar una continuación coherente.

```python
from transformers import pipeline

generator = pipeline(\"text-generation\", model=\"gpt-2\")

resul = generator(\"¿What is machine learning?\", max_length=20)

print(result)

```

#### Parameter

a.  *Definición*

> Estos son valores numéricos internos dentro de un modelo que se
> ejecutan durante la fase de entrenamiento; estos incluyen los pesos y
> sesgos que controlan cómo los modelos procesarán la información.

b.  *Por qué es importante en el aprendizaje automático*

> Los parámetros tienen un impacto directo en la precisión de la
> respuesta del modelo, puesto que estos son usados en la fase de
> entrenamiento, en pocas palabras sin parámetros de entrenamiento no
> habría un modelo fiable.

c.  *Relación con GPT-2 o Transformadores*

> La relación de los parámetros con GPT-2 es que GPT-2 incluye los
> parámetros utilizados para entrenar constantemente el modelo,
> incluidos:
>-   Incrustaciones de palabras
>-   Capas de atención
>-   Capas lineales
>-   Capas de salida

> Cuando usamos GPT-2 con la biblioteca Transformers, lo que estamos
> haciendo es cargar el modelo con todos los parámetros pre-entrenados
> que entienden el conocimiento del lenguaje.

d.  *Ejemplo práctico*

> El siguiente ejemplo muestra una forma de contar la cantidad de
> parámetros que tendrá el modelo que estamos utilizando.

```python
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(\"gpt2\")

total_params = sum(p.numel() for p in model.parameters())

print(total_params)

```

####  Tokenizador

a.  *Definición*

> Un tokenizador es una herramienta que convierte texto en unidades más
> pequeñas llamadas tokens, comúnmente utilizadas para información
> confidencial que un modelo puede eventualmente procesar.

b.  *Por qué es importante en el aprendizaje automático*

> Sin un tokenizador, un modelo de lenguaje no podría funcionar con
> textos, ya que para que la información que ingresamos en el modelo de
> lenguaje sea entendida, tendría que ser convertida en datos numéricos
> y tendría que estar bien estructurada.

c.  *Relación con GPT-2 o Transformadores*

> GPT-2 utiliza un tokenizador basado en subpalabras.
>
> Por otro lado, en Liberia Transformers, cada modelo tiene su propio
> tokenizador, que convierte los datos de entrada en datos numéricos,
> conocidos como IDSInput, y los envía al modelo para generar
> predicciones o texto.

d.  *Ejemplo práctico*

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")

texto = \"Machine learning is powerful\"

# De texto a numeros

tokens = tokenizer.encode(texto)

print(tokens)

# De tokens a texto

decoded_text = tokenizer.decode(tokens)

print(decoded_text)

```

####  Tokens

a.  *Definición*

> Los tokens son las unidades de texto más pequeñas que procesa un
> modelo de lenguaje.

b.  *Por qué es importante en el aprendizaje automático*

> Los modelos de lenguaje no trabajan directamente con texto, sino con
> tokens convertidos en números.

c.  *Relación con GPT-2 o Transformadores*

> GPT-2 divide el texto en pequeñas partes llamadas subpalabras mediante
> un método llamado Codificación de Pares de Bytes (BPE). En la
> biblioteca Hugging Face Transformers, el texto se convierte primero en
> tokens mediante un tokenizador. Luego, cada token se transforma en un
> número (ID de token). Estos números se envían al modelo como entrada,
> y este los utiliza para predecir el siguiente token y así generar
> texto.

d.  *Ejemplo práctico*

```python
from transformers import GPT2Tokenizer

# Cargando tokenizador

tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")

texto = \"Machine learning is amazing\"

# Obtener tokens

tokens = tokenizer.tokenize(texto)

print(tokens)

# Convertir tokens to IDs

token_ids = tokenizer.encode(texto)

print(token_ids)

```

####  Tensor

a.  *Definición*

> Es una estructura de datos multidimensional que contiene números
> utilizados para representar datos y realizar cálculos en modelos de
> aprendizaje automático.

b.  *Por qué es importante en el aprendizaje automático*

> El aprendizaje automático se basa en gran medida en las matemáticas
> aplicadas a grandes volúmenes de datos, y los tensores permiten la
> representación eficiente de estos datos, lo que facilita cálculos
> rápidos de GPU y CPU.

c.  *Relación con GPT-2 o Transformadores*

> Tanto en GPT-2 como en la biblioteca de transformadores, el texto
> ingresado se convierte en tensores numéricos.

d.  *Ejemplo práctico*

```python
from transformers import GPT2Tokenizer

import torch

# Cargar tokenizador

tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")

texto = \"Hello world\"

# Convertir texto a tenzor

inputs = tokenizer(texto, return_tensors=\"pt\")

print(inputs\[\"input_ids\"\])

print(type(inputs\[\"input_ids\"\]))

```

####  PyTorch

a.  *Definición*

> PyTorch es una biblioteca de código abierto desarrollada por Meta para
> crear y entrenar modelos de aprendizaje automático y redes neuronales.

b.  *Por qué es importante en el aprendizaje automático*

> PyTorch es importante porque simplifica la construcción y el
> entrenamiento de modelos de aprendizaje automático. Es fácil de usar,
> flexible y permite visualizar y depurar modelos paso a paso, lo que lo
> hace muy popular para la investigación y el desarrollo.

c.  *Relación con GPT-2 o Transformadores*

> La biblioteca Hugging Face Transformers está diseñada principalmente
> para funcionar con PyTorch. Modelos como GPT-2 se pueden cargar,
> ejecutar y ajustar con PyTorch, donde los datos, pesos y operaciones
> del modelo se gestionan como tensores.

d.  *Ejemplo práctico*

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar tokenizador y modelo

tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")

model = GPT2LMHeadModel.from_pretrained(\"gpt2\")

# Texto de entrada

text = \"Deep learning is\"

inputs = tokenizer(text, return_tensors=\"pt\")

# Genera Texto

outputs = model.generate(inputs\[\"input_ids\"\], max_length=20)

print(tokenizer.decode(outputs\[0\]))

```

####  TensorFlow

a.  *Definición*

> It is a library created by Google to build, train, and run machine
> learning and deep learning models.

b.  *Por qué es importante en el aprendizaje automático*

> TensorFlow es una de las bibliotecas más utilizadas en la actualidad,
> proporciona herramientas para diseñar y entrenar modelos de
> aprendizaje automático, lo que permite trabajar con grandes volúmenes
> de datos y optimizar el rendimiento mediante el uso de hardware
> actualizado.

c.  *Relación con GPT-2 o Transformadores*

> Esto se relaciona principalmente con problemas de dependencia y
> compatibilidad; por ejemplo, la biblioteca Transformers se puede
> utilizar dentro de TensorFlow o Sickitlearn, y, del mismo modo, el
> modelo GPT-2 se puede utilizar dentro de TensorFlow.

d.  *Ejemplo práctico*

```python
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Cargar tokenizador y modelo en tensorflow

tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2\")

model = TFGPT2LMHeadModel.from_pretrained(\"gpt2\")

texto = \"Machine learning is\"

inputs = tokenizer(texto, return_tensors=\"tf\")

# Cargar Texto

outputs = model.generate(inputs\[\"input_ids\"\], max_length=20)

print(tokenizer.decode(outputs\[0\]))

```

####  CUDA (Arquitectura de dispositivo unificado de cómputo)

a.  *Definición*

> Es la plataforma de computación paralela y el modelo de programación
> de NVIDIA que permite a los desarrolladores utilizar GPUs para
> computación de propósito general (GPGPU), no solo para el renderizado
> de gráficos.

b.  *Por qué es importante en el aprendizaje automático*

> La mayoría de los frameworks de deep learning (como PyTorch y
> TensorFlow) dependen de CUDA para acelerar las operaciones con
> tensores.

c.  *Relación con GPT-2 o Transformadores*

> GPT-2 depende en gran medida de grandes multiplicaciones de matrices
> (por ejemplo, en las capas de atención), que se aceleran
> significativamente utilizando GPUs con soporte CUDA.

d.  *Ejemplo práctico*

```python
import torch

device = torch.device(\'cuda\' if torch.cuda.is_available() else
\'cpu\')

model = GPT2LMHeadModel.from_pretrained(\'gpt2\').to(device)

```

#### CPU vs GPU

a.  *Definición*

-   La CPU (Central Processing Unit) está optimizada para tareas
    > secuenciales y de propósito general utilizando un pequeño número
    > de núcleos potentes.

-   La GPU (Graphics Processing Unit) está optimizada para computación
    > paralela utilizando miles de núcleos más pequeños.

b.  *Por qué es importante en el aprendizaje automático*

> Las cargas de trabajo de deep learning son altamente paralelizables
> porque implican operaciones con tensores que pueden calcularse
> simultáneamente.

c.  *Relación con GPT-2 o Transformadores*

> El procesamiento de los mecanismos de atención a través de todos los
> tokens se beneficia enormemente de la paralelización en GPU. Un lote
> (batch) que tarda minutos en CPU puede tardar segundos en GPU.

#### Inference

a.  *Definición*

> Uso de un modelo entrenado para hacer predicciones sobre datos nuevos
> sin actualizar los parámetros del modelo.

b.  *Por qué es importante en el aprendizaje automático*

> La inferencia es lo que permite que los modelos entrenados se utilizan
> en aplicaciones reales como chatbots, sistemas de traducción, motores
> de recomendación y reconocimiento de imágenes.

c.  *Relación con GPT-2 o Transformadores*

> Cuando cargas GPT-2 y generas texto, estás realizando inferencia. Los
> pesos del modelo permanecen congelados; simplemente pasas las entradas
> por la red.

d.  *Ejemplo práctico*

```python
from transformers import pipeline

\# Inferencia: uso del modelo para generar predicciones

generator = pipeline(\'text-generation\', model=\'gpt2\')

output = generator(\"Machine learning is\", max_length=30)

```

#### Training

a.  *Definición*

> Proceso de ajustar los parámetros de un modelo (pesos y sesgos)
> alimentándolo con datos.

b.  *Por qué es importante en el aprendizaje automático*

> El entrenamiento es la forma en que los modelos aprenden patrones a
> partir de los datos. Es computacionalmente costoso, ya que requiere
> muchas iteraciones sobre el conjunto de datos mientras se calculan
> gradientes y se actualizan miles de millones de parámetros.

c.  *Relación con GPT-2 o Transformadores*

> GPT-2 fue entrenado originalmente con un large corpus de texto de
> internet utilizando aprendizaje auto supervisado.

d.  *Ejemplo práctico*

```python
outputs = model(\*\*inputs, labels=inputs\[\"input_ids\"\])

loss = outputs.loss

loss.backward()

optimizer.step()

```

#### Fine-tuning

a.  *Definición*

> El fine-tuning consiste en tomar un modelo pre entrenado y continuar
> su entrenamiento con un conjunto de datos más pequeño y específico
> para adaptarlo a un caso de uso concreto.

b.  *Por qué es importante en el aprendizaje automático*

> En lugar de entrenar desde cero (costoso y requiere grandes cantidades
> de datos), se aprovecha el conocimiento ya aprendido y se especializa.
> Esto permite obtener buenos resultados con muchos menos datos y
> recursos computacionales.

c.  *Relación con GPT-2 o Transformadores*

> Puedes hacer fine-tuning de GPT-2 con textos médicos para aplicaciones
> sanitarias o con código para asistencia en programación. El
> fine-tuning actualiza ligeramente los pesos del modelo para adaptarlo
> a un dominio específico, manteniendo la mayor parte de su conocimiento
> previo.

d.  *Ejemplo práctico*

```python
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

# Load pretrained model

model = GPT2LMHeadModel.from_pretrained(\'gpt2\')

# Fine-tune on custom dataset

training_args = TrainingArguments(output_dir=\'./results\',
num_train_epochs=3)

trainer = Trainer(model=model, args=training_args,
train_dataset=custom_dataset)

trainer.train()

```

#### Pretrained model

a.  *Definición*

> Un modelo pre entrenado es una red neuronal que ya ha sido entrenada
> con un gran conjunto de datos y cuyos parámetros aprendidos se guardan
> y se ponen a disposición para su reutilización.

b.  *Por qué es importante en el aprendizaje automático*

> Los modelos pre entrenados democratizan la IA al proporcionar puntos
> de partida potentes. En lugar de necesitar enormes recursos para
> entrenar desde cero, cualquiera puede usar modelos de última
> generación y adaptarlos.

c.  *Relación con GPT-2 o Transformadores*

> GPT-2 es un modelo de lenguaje basado en Transformers que fue pre
> entrenado y publicado por OpenAI, y se distribuye a través de
> librerías como Hugging Face Transformers.

d.  *Ejemplo práctico*

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained model (already trained on massive text corpus)

tokenizer = GPT2Tokenizer.from_pretrained(\'gpt2\')

model = GPT2LMHeadModel.from_pretrained(\'gpt2\')

# Ready to use immediately - no training needed

text = tokenizer(\"AI is\", return_tensors=\'pt\')

output = model.generate(\*\*text)

```

### Parte 2 --- Reflexión práctica

#### ¿Cuál es la diferencia entre entrenamiento e inferencia?

>El entrenamiento es el proceso en el cual un modelo aprende de los
>datos, mientras que la inferencia es el proceso en el cual se utiliza un
>modelo que ya está entrenado para generar predicciones o respuestas.

#### ¿Por qué usar una GPU (CUDA) acelera el entrenamiento?

>CUDA es una plataforma desarrollada por NVIDIA que permite que
>bibliotecas como PyTorch y TensorFlow envíen más datos a la GPU, lo que
>aumenta el rendimiento.

#### ¿Qué sucede internamente cuando llamas a pipeline(\"text-generation\")?

```python
from transformers import pipeline

generator = pipeline(\"text-generation\")
```

>Primero, se selecciona un modelo adecuado para generar texto; este
>suele ser GPT-2, a menos que se especifique lo contrario. A
>continuación, se descarga y carga el tokenizador, que convierte el texto
>en números que el modelo puede comprender. A continuación, se descarga y
>carga el modelo, junto con sus pesos preentrenados que contienen el
>conocimiento del lenguaje. Finalmente, se crea un objeto que combina el
>modelo y el tokenizador, listo para usar.

#### ¿Por qué podemos usar GPT-2 sin entrenarlo nosotros mismos?

>Podemos usar GPT-2 nosotros mismos sin necesidad de preentrenarlo, ya
>que OpenAI ya lo ha entrenado con grandes cantidades de texto.

### Parte 3 --- Reflexión práctica
#### ¿Son los pesos o weights descargados siempre? Por que o Por que no?

>No, los weights no se descargan cada vez que corro el modelo solo ocurre
>la primera vez, ya que la librería transformers los guarda en caché.

#### ¿Que cambios hay al pasar de gpt2 a distilgpt2?

>El primer cambio aparece en la velocidad el modelo distilgpt2 es mucho
>más rápido que el modelo gpt2. La segunda diferencia importante es en
>términos de peso ya que el modelo gpt2 pesa 523 mb, mientras que el
>modelo distilgpt2 pesa 337 mb.

#### ¿El script realiza entrenamiento o inferencia?

>El script está realizando una inferencia, porque estamos usando el
>modelo al hacer prompts o preguntar cosas, por ende no estamos
>entrenando o dando contexto al modelo usando data. Además para poder
>entrenar un modelo se requiere de una tarjeta gráfica potente.

#### ¿Por qué la generación tarda más en la CPU en comparación con la GPU?

>Toma más tiempo en la CPU porque está diseãdo para realizar
>procesamiento secuencial, en cambio las GPUs están pensadas para
>realizar varias tareas a la vez(procesamiento en paralelo)

### Bonus (Opcional)

#### Descubre cuantos parámetros tiene GPT2.

El modelo tiene 124,439,808 parámetros

![Imagen 4](https://github.com/MarioJGC/repository/blob/main/image4.png)

#### Comparación GPT-2 small vs GPT-2 medium.

| Característica/Modelo | GPT2-SMALL | GPT2-MEDIUM |
|--------|------|--------|
| Velocidad | Mas rapido | Más lento |
| Número de parámetros | 124,439,808 | 354,823,168 |
| Peso | 523 mb | 1.5 gb |


#### Explica el significado de "next-token prediction"

Consiste en que en base al prompt que usamos en el modelo este realizará
una predicción de lo que puede seguir después por ejemplo en la prueba
realizada en GPT-2, yo coloque lo siguiente como prompt: give me fruits,
inmediatamente el modelo me respondió lo siguiente give me fruits from
my mother. En conclusión auto completa mi prompt con el token que podría
seguir en base a una probabilidad y lo vuelve a repetir hasta terminar
su respuesta.

![Imagen 3](https://github.com/MarioJGC/repository/blob/main/image3.png)

## Comparación GPT-2 y Qwen3 1.7B
### Parte 1: Ejecutar ambos modelos
>1. Ejecutar la CLI de GPT-2.
>2. Ejecutar la CLI de Qwen3 1.7B.
>3. Intente la misma entrada en ambos modelos y observe qué sucede.
>Familiarícese con ambas herramientas antes de continuar.

### Parte 2: Diseña tus propios experimentos
Experimento: ¿ Pueden los dos modelos seguir instrucciones ?

Voy a preguntar a cado uno de los modelos "Escribeme una lista de pasos
a seguir en caso de un huracán"

#### 1\. Hipótesis

>**GPT2** - No va a ni escribir los pasos. Va a escribir algo relacionado
>con huracanes, pero no va a tener mucho que ver con la tarea, y por
>supuesto no la va a cumplir.

>**Qwen3** - Va a escribir los pasos. Quizá no sean perfectos, pero va a
>ser una respuesta coherente y alineada con la tarea.

#### 2\. Prompts

>Prompt: "Escríbeme una lista de pasos a seguir en caso de un huracán".

>Es un prompt simple diseñado para ver si cada modelo es capaz de
>entender el prompt y retornar una respuesta que 1) sea coherente, 2)
>cumpla con la tarea.

#### 3\. Mostrar los outputs

>**GPT2**
![Imagen 1](https://github.com/MarioJGC/repository/blob/main/image1.png)

>**Qwen3 1.7B**
![Imagen 2](https://github.com/MarioJGC/repository/blob/main/image2.png)

#### 4\. Analiza los resultados: ¿Coincidieron con tus expectativas?

Como se puede observar, GPT2 produjo una respuesta que tenía poco que
ver con la tarea. Le pedí una lista de pasos a seguir en caso de un
huracán, y me respondió con "Mi esposo ya me ha dicho todo lo que sabe
de los huracanes"

Mientras que Qwen3 si cumplio con la tarea. No creo que sea comprensiva
la respuesta, es decir que podría mejorar, pero aún así cumplió con la
tarea. "Step 1 .... Step 2 .... ". Es una respuesta que tiene más
sentido según lo pedido.

### Parte 3 --- Reflexión técnica

#### Conecta tus hallazgos experimentales con estos conceptos. Usa ejemplos específicos de tus resultados como evidencia.

>"GPT-2 está entrenado únicamente para next token prediction en texto
>web."

>Según el experimento esto tiene sentido, GPT-2 respondió a mi prompt
>pidiendo pasos en caso de un huracán como si fuera un cuento. Iba
>continuando el cuento en vez de reconocer la estructura del prompt y
>darle una respuesta que tuviera sentido.

>"Qwen3 is instruction-tuned, preference-aligned, and supports a
>reasoning mode."

#### ¿Que es instruction tuning? 
>El ajuste de instrucciones es el proceso de
>tomar un modelo base y entrenarlo aún más en un conjunto de datos muy
>específico formateado como pares to (instrucción -\> respuesta)

#### ¿Que es alignment?
>Alineación es el paso en el que se le enseñan al
>modelo las preferencias humanas - específicamente a ser útil y
>estructurado

#### ¿Cómo el fine-tuning cambia el comportamiento del modelo sin modificar su arquitectura?
>Dado que el fine-tuning permite obtener resultados
>diferentes, en ocasiones mejores sin necesidad de cambiar la
>arquitectura del modelo o en este caso entrenarlo nuevamente. Las
>probabilidades de GPT-2 favorecen las continuaciones aleatorias tipo
>internet. En cambio el fine-tuning de Qwen3 cambió matemáticamente sus
>probabilidades para favorecer en gran medida las respuestas
>estructuradas tipo asistente cada vez que ve un prompt humano.

>Según los resultados, podemos observar que QWEN fue capaz de reconocer
>el prompt y formatear una respuesta con sentido. Es decir, debido a que
>pasó por el ajuste de instrucciones, reconoció la forma estructural del
>prompt como un comando a ejecutar, y por lo tanto, [cambió su
>comportamiento como narrador a un asistente.]{.underline} Su intento de
>darme una lista clara, numerada y organizada muestra que está alineado
>para formatear la información de una manera que los humanos consideran
>útil.

### Parte 4 --- Conclusiones

#### ¿Que revelaron tus experimentos sobre el rol training strategy?

>Los experimentos revelaron que la estrategia de entrenamiento, y no solo
>el tamaño del modelo, es el verdadero motor de utilidad. El
>preentrenamiento (lo que tuvo GPT-2) simplemente construye un mapa
>estadístico del lenguaje. Le enseña al modelo como hablar, pero no que
>es apropiado decir. La estrategia de entrenamiento de ajuste de
>instrucciones y alineación (lo que tuvo Qwen3) es lo que realmente
>transforma un predictor de texto crudo en una herramienta utilizable que
>comprende la intención y formatea las respuestas de manera útil.

#### ¿Pueden dos modelos de arquitectura similar comportarse de forma diferente? Por que?

>Si, dos modelos se pueden comportar de manera diferente incluso si
>comparten exactamente la misma arquitectura Transformer. Ambos modelos
>procesan las entradas para predecir el siguiente token. Sin embargo, el
>fine-tuning cambia los pesos (las probabilidades) del modelo. Los pesos
>de GPT-2 favorecen la predicción de la siguiente palabra más común
>encontrada en internet, los pesos de Qwen3 han sido alterados
>matemáticamente para favorecer la predicción de la siguiente palabra de
>una respuesta útil y estructurada.

#### ¿Qué te sorprendió más?

>Lo que más me sorprendió fueron las diferencias en las respuestas. GPT-2
>no fue capaz de reconocer y "entender" el prompt, por así decirlo,
>simplemente respondía siguiendo la corriente del prompt, mientras que
>Wen3 sí reconoció y entendió la tarea, lo que es evidente en su
>respuesta con una lista numerada de pasos.

>También nos sorprendió el hecho de que a pesar de tener modelos con
>arquitecturas establecidas podemos ajustarlos de tal manera que logren
>ser más eficientes como el caso de Qwen3 que tiene un fine-tuning
>especial para absolver consultas, lo cual le da ventaja frente a modelos
>que no lo contemplan como fue el caso de GPT2

#### ¿Qué probarías a continuación si tuvieras más tiempo?

>En este ejercicio experimentamos la capacidad de los dos modelos de
>entender la tarea. Lo que haría si tuviera más tiempo es jugar un poco
>con los prompts. En vez de darle a los modelos una tarea específica y
>bien escrita, qué pasaría si le escribiera un prompt mal escrito, con
>muchas palabras mal escritas, pero que tuviera algo de sentido. ¿Serían
>capaces de discernir las partes importantes?

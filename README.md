# Extracción de Información de CVs en PDF

## Descripción

Este notebook tiene como objetivo extraer información clave de los currículos (CVs) en formato PDF. Utiliza técnicas de procesamiento de lenguaje natural (NLP) y expresiones regulares para identificar y extraer datos relevantes, como nombre, contacto, experiencia y educación en inteligencia artificial (IA). Los resultados se comparan con valores de referencia para evaluar la precisión del proceso.

## Requisitos

Instala las siguientes librerías antes de ejecutar el notebook:

```bash
!pip install transformers
!pip install torch
!pip install nltk
!pip install spacy
!python -m spacy download en_core_web_sm
!pip install pandas
!pip install pymupdf
```

## Importación de Librerías

```python
import fitz  # PyMuPDF
import os
import re
import json
from transformers import pipeline
import spacy
```

## Configuración del Entorno

Define el directorio donde se encuentran los archivos PDF de los CVs y lista todos los archivos PDF en el directorio.

```python
cv_dir = "/content/drive/MyDrive/IA/CVs"
cv_files = [os.path.join(cv_dir, file) for file in os.listdir(cv_dir) if file.endswith('.pdf')]
```

## Funciones de Extracción

### Extracción de Texto desde PDF

```python
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text += page.get_text()
    return text
```

Extrae el texto de todos los archivos PDF y guarda el texto junto con el nombre del archivo.

```python
cv_texts = [(file, extract_text_from_pdf(file)) for file in cv_files]
```

### Valores de Referencia

Define un conjunto de valores de referencia para comparar los resultados extraídos.

```python
ground_truth = [
    {"file": cv_files[0], "name": "IMMANUEL ABRAHAM MAHARDHIKA", "contact": ["+62 8577 7124 773", "dhikayudano@gmail.com"], "experience": 7, "ai_education": "N"},
    # ... (otros valores de referencia)
]
```

### Carga de Modelos NLP

Carga modelos NLP para Named Entity Recognition (NER) y análisis de texto.

```python
nlp_ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")
nlp_spacy = spacy.load("en_core_web_sm")
```

### Funciones de Extracción

#### Extracción de Nombre

```python
def extract_name(text):
    entities = nlp_ner(text)
    names = [entity['word'] for entity in entities if entity['entity'] == 'B-PER']
    return " ".join(names) if names else None
```

#### Extracción de Contacto

```python
def extract_contact(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b\d{10,15}\b'
    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    return email.group() if email else phone.group() if phone else None
```

#### Extracción de Experiencia

```python
def extract_experience(text):
    experience_keywords = ["years of experience", "experience", "years"]
    experience = None
    for keyword in experience_keywords:
        match = re.search(r'\d+', text)
        if match:
            experience = int(match.group())
            break
    return experience
```

#### Extracción de Educación en IA

```python
def extract_ai_education(text):
    ai_keywords = ["artificial intelligence", "machine learning", "deep learning", "AI"]
    text_lower = text.lower()
    return "S" if any(keyword.lower() in text_lower for keyword in ai_keywords) else "N"
```

### Crear Resultado en Formato JSON

```python
def create_result_json(file, text):
    gt = find_ground_truth(file)
    name = extract_name(text)
    contact = extract_contact(text)
    experience = extract_experience(text)
    ai_education = extract_ai_education(text)

    result = {
        "name": name,
        "contact": contact,
        "experience": experience,
        "ai_education": ai_education,
        "name_score": calculate_score(name, gt.get('name')),
        "contact_score": calculate_score(contact, gt.get('contact')),
        "experience_score": calculate_score(experience, gt.get('experience')),
        "ai_education_score": calculate_score(ai_education, gt.get('ai_education'))
    }
    return json.dumps(result, indent=4)
```

## Aplicación de las Funciones y Resultados

Aplica las funciones de extracción a los textos obtenidos de los CVs y muestra los resultados.

```python
results = [(file, create_result_json(file, text)) for file, text in cv_texts]

# Mostrar resultados (opcional)
for result in results:
    print("########################################")
    print(result[0] + ":")
    print(result[1])
    print()
    print("########################################")
```
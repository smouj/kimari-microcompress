<div align="center">

# Kimari MicroCompress

**Compresión sin pérdidas reversible para archivos de modelos de IA**

*Safetensors · GGUF · LoRA/PEFT · Checkpoints de entrenamiento · Modelos de Hugging Face*

[![CI](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml/badge.svg)](https://github.com/smouj/kimari-microcompress/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.7.0--alpha-orange.svg)](https://github.com/smouj/kimari-microcompress/releases)
[![Tests](https://img.shields.io/badge/tests-228%20passing-brightgreen.svg)](https://github.com/smouj/kimari-microcompress/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-DB7093.svg)](https://docs.astral.sh/ruff/)

</div>

---

## ⚠️ Limitaciones importantes

> **Por favor, lea antes de usar KMC:**

- **KMC NO reduce la VRAM durante la inferencia.** Está diseñado para almacenamiento, transferencia y verificación — no para optimización de memoria en tiempo de ejecución.
- **KMC NO modifica los pesos del modelo.** La compresión es sin pérdidas y reversible; cada byte se preserva exactamente.
- **La carga por bloques (descompresión parcial) es investigación futura.** El formato almacena bloques con offsets, pero el servidor de bloques bajo demanda aún no está implementado.
- **La compresión con conocimiento de GGUF es experimental.** El flag `--gguf-aware` ajusta la selección de códec para tensores GGUF cuantizados, pero aún no implementa estrategias de compresión específicas de GGUF a nivel de bloque.
- **No se deben asumir ratios de compresión fijos.** Los resultados varían significativamente según el formato del modelo, el tipo de dato y el contenido. Los benchmarks sintéticos no representan ratios del mundo real.
- **KMC no es un reemplazo para la cuantización.** Si necesita modelos más pequeños para inferencia, use cuantización (GGUF Q4_K, GPTQ, AWQ, etc.). KMC es complementario: comprime archivos ya cuantizados para almacenamiento/transferencia.
- **No se usa pickle.** KMC nunca deserializa archivos basados en pickle. Solo se registran presencia, tamaño y hash.
- **KMC es únicamente sin pérdidas.** No existe un modo con pérdidas ni modificación de pesos de ningún tipo.

### ⚠️ Advertencias sobre acceso parcial (v0.7.0-alpha)

> **Importante — comprenda estas limitaciones antes de usar las funciones de acceso parcial:**

- **KMC no realiza inferencia comprimida.** Las funciones de acceso parcial extraen datos del archivo; no ejecutan el modelo ni cargan tensores en GPU.
- **La carga parcial de tensores devuelve bytes a menos que se instalen librerías opcionales.** Sin dependencias como `safetensors` o `torch`, `kmc extract --tensor` devuelve los bytes en bruto del tensor, no un tensor utilizable.
- **La extracción de tensores depende de metadatos capturados durante el empaquetado.** Si un archivo se empaquetó sin `--tensor-aware`, los nombres de tensores y offsets pueden no estar disponibles para extracción selectiva.
- **Archivos antiguos pueden soportar acceso parcial a nivel de archivo pero no a nivel de tensor.** Los archivos `.kmc` creados con versiones anteriores a v0.7.0 pueden tener índices de archivo pero carecen de índices de tensor granulares necesarios para `--tensor`.

---

## Tabla de contenidos

- [Descripción general](#descripción-general)
- [Características](#características)
- [Instalación](#instalación)
- [Inicio rápido](#inicio-rápido)
- [Referencia de CLI](#referencia-de-cli)
- [Códecs](#códecs)
- [Formato de archivo](#formato-de-archivo)
- [Arquitectura](#arquitectura)
- [Seguridad](#seguridad)
- [Documentación](#documentación)
- [Desarrollo](#desarrollo)
- [Fundamento técnico](#fundamento-técnico)
- [Licencia](#licencia)

---

## Descripción general

Kimari MicroCompress (KMC) es una herramienta experimental para la **compresión sin pérdidas y reversible** de archivos de modelos de IA. Se enfoca en **almacenamiento, transferencia, verificación y empaquetado** sin modificar los pesos originales. El enfoque se basa en la observación de que los archivos de modelos de IA — particularmente `safetensors` y formatos cuantizados — contienen redundancia significativa que las herramientas de compresión de propósito general no explotan de manera óptima.

**Principio clave:** Cada byte que entra debe salir idénticamente. KMC proporciona integridad de ida y vuelta exacta en bytes verificada mediante hashes SHA-256 tanto a nivel de archivo como a nivel de bloque.

### ¿Por qué KMC?

| Problema | Solución de KMC |
|----------|-----------------|
| Los archivos de modelos de IA son grandes y costosos de almacenar | Compresión sin pérdidas con códecs conscientes de tensores (BytePlane, FloatPlane) |
| Las herramientas de propósito general ignoran la estructura de tensores | Selección de códec consciente del dtype por bloque (FP32, BF16, FP16, cuantizados) |
| Sin garantías de integridad después de la compresión | Verificación SHA-256 a nivel de archivo y de bloque |
| Artefactos mixtos (modelo + LoRA + checkpoints) | Detección de tipo de artefacto y flujos de trabajo especializados |
| Los datos cuantizados GGUF no se comprimen bien | Modo `--gguf-aware` experimental que adapta la selección de códec |
| Sin visibilidad del contenido de un archivo | `kmc inspect` con metadatos específicos del formato y detalles de tensores |

---

## Características

| Característica | Estado |
|----------------|--------|
| `kmc pack` — Comprimir archivos/directorios | ✅ Funcional |
| `kmc pack --tensor-aware` — Alineación de bloques consciente de tensores | ✅ Funcional |
| `kmc pack --gguf-aware` — Compresión con conocimiento de GGUF experimental | 🧪 Experimental |
| `kmc pack-lora` — Flujo de trabajo para adaptadores LoRA | ✅ Funcional |
| `kmc pack-checkpoint` — Flujo de trabajo para checkpoints de entrenamiento | ✅ Funcional |
| `kmc unpack` — Descomprimir archivos (seguro ante path traversal) | ✅ Funcional |
| `kmc verify` — Informe completo de verificación | ✅ Funcional |
| `kmc inspect` — Inspección de modelos de IA con metadatos de tensores | ✅ Funcional |
| `kmc inspect --json` — Salida JSON para scripting | ✅ Funcional |
| `kmc inspect --tensors` — Información detallada de tensores | ✅ Funcional |
| `kmc inspect --lora` — Inspección de adaptador LoRA | ✅ Funcional |
| `kmc inspect --checkpoint` — Inspección de checkpoint de entrenamiento | ✅ Funcional |
| `kmc inspect --gguf` — Inspección de modelo GGUF con detalles de tensores | ✅ Funcional |
| `kmc bench` — Benchmark con comparación de códecs | ✅ Funcional |
| `kmc bench --compare-codecs` — Comparación de múltiples códecs | ✅ Funcional |
| `kmc bench --compare-zipnn` — Comparación con ZipNN | ✅ Funcional |
| Detección automática de artefactos (HuggingFace, GGUF, LoRA, checkpoint) | ✅ Funcional |
| Analizador de metadatos de tensores GGUF (nombres, formas, tipos, offsets, tamaños) | ✅ Funcional |
| Resumen de cuantización GGUF (Q4_K, Q5_1, F32, etc.) | ✅ Funcional |
| Manifest v4 con artifact_type, artifact_metadata, format_metadata | ✅ Funcional |
| Hashing SHA-256 por archivo y por bloque | ✅ Funcional |
| Micro-bloques de 256 KiB (configurable) | ✅ Funcional |
| Selección de códec zstd / zlib / raw / byteplane / floatplane | ✅ Funcional |
| Selector de códec automático (basado en dtype) | ✅ Funcional |
| Metadatos reales de tensores safetensors (nombres, formas, dtypes, offsets) | ✅ Funcional |
| Detección de adaptadores LoRA/PEFT con rank y módulos objetivo | ✅ Funcional |
| Protección contra path traversal en unpack | ✅ Funcional |
| Compatible hacia atrás con .kmc v0.2/v0.3/v0.4 | ✅ Funcional |
| Compresión GGUF a nivel de bloque | 🔬 Investigación |
| Carga por bloques (descompresión parcial) | 🔬 Investigación |

### Novedades en v0.7.0-alpha: Acceso parcial y extracción selectiva

La versión v0.7.0-alpha introduce capacidades de acceso parcial, permitiendo extraer componentes específicos de un archivo `.kmc` sin necesidad de descomprimirlo por completo.

| Característica | Estado |
|----------------|--------|
| Índices de bloques/archivos/tensores dentro del archivo `.kmc` | ✅ Funcional |
| API `KMCReader` para acceso parcial (lectura sin descompresión completa) | ✅ Funcional |
| Extracción selectiva con `--only`, `--tensor`, `--list` | ✅ Funcional |
| Comando `kmc list` para listar contenido del archivo | ✅ Funcional |
| Cargador experimental de safetensors desde archivos `.kmc` | 🧪 Experimental |
| Benchmarks de acceso parcial | ✅ Funcional |

**Ejemplos de acceso parcial:**

```bash
# Listar el contenido de un archivo sin descomprimir
kmc list ./my-model.kmc

# Extraer solo archivos específicos
kmc unpack ./my-model.kmc ./output/ --only "model.safetensors" --only "config.json"

# Extraer un tensor específico por nombre
kmc extract ./my-model.kmc --tensor "transformer.h.0.attn.c_attn.weight"

# Listar tensores disponibles en el archivo
kmc extract ./my-model.kmc --list

# Usar KMCReader mediante la API de Python
python -c "
from kmc.reader import KMCReader
with KMCReader('./my-model.kmc') as r:
    print(r.list_files())
    print(r.list_tensors())
    data = r.read_tensor('transformer.h.0.attn.c_attn.weight')
    print(f'Tensor bytes: {len(data)}')
"
```

---

## Instalación

```bash
# Clonar e instalar en modo de desarrollo
git clone https://github.com/smouj/kimari-microcompress.git
cd kimari-microcompress
pip install -e ".[dev]"

# Con dependencia opcional safetensors (análisis mejorado de cabeceras)
pip install -e ".[safetensors]"

# Con dependencia opcional ZipNN (para comparación de benchmarks)
pip install -e ".[zipnn]"

# Todas las dependencias opcionales
pip install -e ".[all]"
```

### Requisitos

| Dependencia | Requerida | Propósito |
|-------------|-----------|-----------|
| Python 3.10+ | Sí | Tiempo de ejecución |
| `zstandard` | Sí | Mejor códec de compresión |
| `zlib` | Sí (integrado) | Códec de compresión alternativo |
| `safetensors` | No (opcional) | Análisis mejorado de cabeceras safetensors |
| `zipnn` | No (opcional) | Comparación de benchmarks |

---

## Inicio rápido

```bash
# Empaquetar un directorio de modelo
kmc pack ./my-model ./my-model.kmc

# Empaquetar con modo consciente de tensores (recomendado para safetensors)
kmc pack ./my-model ./my-model.kmc --tensor-aware

# Empaquetar con modo consciente de GGUF (experimental, para archivos GGUF)
kmc pack ./my-model ./my-model.kmc --gguf-aware

# Empaquetar un adaptador LoRA
kmc pack-lora ./my-lora-adapter ./my-lora.kmc

# Empaquetar un checkpoint de entrenamiento
kmc pack-checkpoint ./checkpoint-1000 ./checkpoint-1000.kmc

# Verificar integridad (informe completo)
kmc verify ./my-model.kmc

# Inspeccionar el manifest del archivo
kmc inspect ./my-model.kmc

# Inspeccionar un directorio de modelo de IA (detecta formatos, lee metadatos de tensores)
kmc inspect ./my-model/ --tensors

# Inspeccionar como adaptador LoRA
kmc inspect ./my-lora/ --lora

# Inspeccionar como checkpoint de entrenamiento
kmc inspect ./checkpoint-1000/ --checkpoint

# Inspeccionar archivo GGUF con detalles de tensores
kmc inspect ./model.gguf --gguf

# Inspeccionar con salida JSON
kmc inspect ./my-model/ --json

# Desempaquetar a un directorio
kmc unpack ./my-model.kmc ./restored-model/

# Ejecutar benchmark con comparación de códecs
kmc bench ./my-model ./my-model-bench.kmc --compare-codecs

# Benchmark con comparación ZipNN
kmc bench ./my-model ./my-model-bench.kmc --compare-zipnn --json --output report.json
```

---

## Referencia de CLI

### Comandos principales

| Comando | Descripción |
|---------|-------------|
| `kmc pack SOURCE OUTPUT` | Comprimir un directorio/archivo en un archivo `.kmc` |
| `kmc pack-lora SOURCE OUTPUT` | Comprimir un directorio de adaptador LoRA |
| `kmc pack-checkpoint SOURCE OUTPUT` | Comprimir un directorio de checkpoint de entrenamiento |
| `kmc unpack ARCHIVE OUTPUT` | Descomprimir un archivo `.kmc` |
| `kmc verify ARCHIVE` | Informe completo de verificación de integridad |
| `kmc inspect TARGET` | Inspeccionar archivo o directorio de modelo de IA |
| `kmc bench SOURCE OUTPUT` | Evaluar el rendimiento de compresión |
| `kmc list ARCHIVE` | Listar el contenido de un archivo `.kmc` |
| `kmc extract ARCHIVE` | Extraer componentes selectivamente del archivo |

### Flags principales

| Flag | Comando | Descripción |
|------|---------|-------------|
| `--tensor-aware` | pack | Alinear bloques a los límites de tensores para archivos safetensors |
| `--gguf-aware` | pack | Ajustar selección de códec para tensores GGUF cuantizados |
| `--codec` | pack, bench | Códec: `auto`, `byteplane`, `floatplane`, `zstd`, `zlib`, `raw` |
| `--lora` | inspect | Inspeccionar como adaptador LoRA |
| `--checkpoint` | inspect | Inspeccionar como checkpoint de entrenamiento |
| `--gguf` | inspect | Inspeccionar como modelo GGUF con detalles de tensores |
| `--tensors` | inspect | Mostrar información detallada de tensores |
| `--compression` | inspect | Mostrar resumen de compresión con uso de códecs |
| `--json` | inspect, bench | Salida como JSON |
| `--compare-codecs` | bench | Comparar todos los códecs disponibles |
| `--compare-zipnn` | bench | Comparar con ZipNN (si está instalado) |
| `--only` | unpack, extract | Extraer solo los archivos especificados |
| `--tensor` | extract | Extraer un tensor específico por nombre |
| `--list` | extract | Listar tensores disponibles en el archivo |

---

## Códecs

KMC v0.7 soporta seis códecs, seleccionados por bloque para resultados óptimos:

| Códec | Tipo | Mejor para | Descripción |
|-------|------|------------|-------------|
| `auto` | Selector | Uso general | Prueba candidatos por dtype, elige el resultado más pequeño |
| `floatplane` | Consciente de tensores | FP32/BF16/FP16 | Separación a nivel de bits de signo/exponente/mantisa |
| `byteplane` | Consciente de tensores | FP32/BF16/FP16 | Separación por planos de bytes según posición dentro del elemento |
| `zstd` | General | Datos mixtos | Compresión de propósito general de alta ratio |
| `zlib` | General | Alternativo | Siempre disponible, compresión decente |
| `raw` | Paso directo | Incompresible | Sin compresión, se usa cuando la compresión expande los datos |

### Selección automática de códec

Cuando se usa `--codec auto` (predeterminado), el selector elige por bloque basándose en el dtype del tensor:

| dtype del tensor | Cadena de candidatos |
|-----------------|---------------------|
| FP32, BF16, FP16 | `floatplane → byteplane → zstd → zlib → raw` |
| Cuantizado (Q4_K, Q8_0, etc.) | `zstd → zlib → raw` |
| Desconocido / no flotante | `zstd → zlib → raw` |

Con `--gguf-aware`, los tensores GGUF cuantizados omiten automáticamente las transformaciones conscientes de flotantes.

---

## Formato de archivo

El formato `.kmc` está diseñado para almacenamiento verificable y orientado a bloques:

```
┌─────────────────────────────────────────┐
│  Magic: "KMC\x00\x01\x00\x00\x00"  8B │
├─────────────────────────────────────────┤
│  Manifest length: uint64 BE         8B │
├─────────────────────────────────────────┤
│  Manifest: JSON (UTF-8)        Variable│
│   - versión, información de herramienta│
│   - entradas de archivo con rutas y hashes │
│   - entradas de bloque con códecs      │
│   - codec_metadata por bloque (v3+)    │
│   - entradas de tensor (v2+, opcional) │
│   - artifact_type (v4+)                │
│   - artifact_metadata (v4+)            │
│   - format_metadata (v4+)              │
├─────────────────────────────────────────┤
│  Block data: concatenados       Variable│
│   - Cada bloque comprimido independientemente │
│   - Verificados con SHA-256 por bloque │
└─────────────────────────────────────────┘
```

Consulte [FORMAT_SPEC.md](docs/FORMAT_SPEC.md) para la especificación completa.

---

## Arquitectura

```
src/kmc/
├── archive.py              # Pack/unpack/verify principal con verificaciones de seguridad
├── benchmark.py            # Benchmarking de rendimiento con comparación de códecs
├── cli.py                  # Interfaz de línea de comandos
├── hashing.py              # Hashing de integridad SHA-256
├── inspector.py            # Detección de formato de modelos de IA con metadatos
├── manifest.py             # Manifest KMC (v4: artifact_type, format_metadata)
├── reader.py               # API KMCReader para acceso parcial (v0.7+)
├── gguf.py                 # Módulo GGUF heredado (ver formats/gguf.py)
├── tensor_inspector.py     # Metadatos safetensors heredados (ver formats/)
├── codecs/
│   ├── __init__.py         # API pública de códecs
│   ├── base.py             # Protocolo de códec, CodecContext, CodecResult
│   ├── byteplane.py        # Códec BytePlane (separación por planos de bytes)
│   ├── floatplane.py       # Códec FloatPlane (separación signo/exp/mantisa)
│   ├── registry.py         # Registro de códecs (descubrir/instanciar por nombre)
│   ├── selector.py         # Selector automático de códec (candidatos basados en dtype)
│   ├── legacy.py           # API CodecId/compress_block heredada (compat v0.2/v0.3)
│   ├── raw.py              # Códec de paso directo raw
│   ├── zlib_codec.py       # Códec zlib
│   └── zstd_codec.py       # Códec zstd
├── formats/
│   ├── __init__.py         # Registro del módulo de formatos
│   ├── safetensors.py      # Metadatos safetensors, shards, detección LoRA
│   └── gguf.py             # Cabecera GGUF + análisis de metadatos de tensores (v0.5+)
├── workflows/
│   ├── __init__.py         # Registro del módulo de flujos de trabajo
│   ├── lora.py             # Detección y empaquetado de adaptadores LoRA/PEFT
│   └── checkpoint.py       # Detección y empaquetado de checkpoints de entrenamiento
└── integrations/
    └── kimari.py           # Adaptadores de integración con Kimari CLI
```

Consulte [ARCHITECTURE.md](docs/ARCHITECTURE.md) para decisiones de diseño detalladas.

---

## Seguridad

KMC toma en serio la seguridad de extracción:

- **Protección contra path traversal** — Todas las rutas de archivo se validan antes de la extracción; se rechazan `..`, rutas absolutas, bytes nulos y caracteres de control
- **Protección contra symlinks** — Rehusa sobreescribir symlinks existentes durante unpack
- **Detección de rutas duplicadas** — Los manifests con rutas de archivo duplicadas son rechazados
- **Límites de tamaño del manifest** — Manifests de tamaño excesivo son rechazados para prevenir DoS
- **Verificación de hash de bloques** — Cada bloque verificado contra su hash SHA-256
- **Verificación de hash de archivos** — Los archivos reconstruidos se verifican contra su hash SHA-256
- **Sin deserialización de pickle** — Los archivos basados en pickle se detectan y comprimen solo como bytes en bruto

Consulte [SECURITY_MODEL.md](docs/SECURITY_MODEL.md) para el modelo de seguridad completo.

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [Arquitectura](docs/ARCHITECTURE.md) | Decisiones de diseño y estructura de módulos |
| [Especificación de formato](docs/FORMAT_SPEC.md) | Especificación completa del formato `.kmc` (v4) |
| [Modelo de seguridad](docs/SECURITY_MODEL.md) | Modelo de amenazas y mitigaciones |
| [Soporte GGUF](docs/GGUF_SUPPORT.md) | Análisis GGUF y modo `--gguf-aware` |
| [Flujo de trabajo LoRA](docs/LORA_WORKFLOW.md) | Compresión e inspección de adaptadores LoRA |
| [Flujo de trabajo de Checkpoints](docs/CHECKPOINT_WORKFLOW.md) | Compresión e inspección de checkpoints de entrenamiento |
| [Flujo de trabajo Hugging Face](docs/HUGGINGFACE_WORKFLOW.md) | Trabajando con modelos de Hugging Face |
| [Benchmark con modelos reales](docs/REAL_MODEL_BENCHMARK.md) | Ejecución de benchmarks con modelos de HuggingFace |
| [Integración con Kimari](docs/KIMARI_INTEGRATION.md) | Integración con Kimari CLI |
| [Plan de benchmarks](docs/BENCHMARK_PLAN.md) | Estrategia de pruebas de rendimiento |
| [Notas de investigación](docs/RESEARCH_NOTES.md) | Referencias técnicas y racional de diseño de códecs |
| [Hoja de ruta](docs/ROADMAP.md) | Prioridades de desarrollo |
| [Registro de cambios](CHANGELOG.md) | Historial de versiones |

---

## Desarrollo

```bash
# Instalar con dependencias de desarrollo
pip install -e ".[dev]"

# Ejecutar pruebas
pytest -q

# Lint
ruff check .

# Verificación de formato
ruff format --check .

# Ayuda de CLI
kmc --help
kmc pack-lora --help
kmc pack-checkpoint --help

# Crear modelo de demostración y probar
python scripts/create_demo_model.py
```

Consulte [CONTRIBUTING.md](CONTRIBUTING.md) para las guías de contribución.

---

## Fundamento técnico

El enfoque de KMC está fundamentado en investigación y práctica de la industria:

- **ZipNN** (IBM Research) — Demuestra que la compresión sin pérdidas específica para modelos de IA puede ahorrar ~1/3 del tamaño en modelos populares, y >50% en algunos casos, sin cambiar los pesos.
- **safetensors** (Hugging Face) — Tratado como el formato prioritario porque es seguro, rápido y evita vulnerabilidades de `pickle`.
- **GGUF** (llama.cpp) — El formato binario estándar para modelos cuantizados. KMC v0.5 añade análisis completo de metadatos de tensores y compresión experimental con conocimiento de GGUF.
- **NetZIP** (IBM Research) — Explora la compresión sin pérdidas para gradientes y activaciones en entrenamiento distribuido — una dirección de investigación documentada en la hoja de ruta de KMC.

---

## Licencia

Licencia MIT — consulte [LICENSE](LICENSE) para más detalles.

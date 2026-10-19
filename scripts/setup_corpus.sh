#!/bin/bash
# ==============================================================================
# Descarga del corpus UD Spanish-AnCora
# ==============================================================================
# Descarga el corpus Universal Dependencies Spanish-AnCora desde GitHub
# y lo coloca en data/corpus/UD_Spanish-AnCora/
#
# Uso:
#   bash scripts/setup_corpus.sh
# ==============================================================================

set -e  # Salir en caso de error

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Descarga del Corpus UD Spanish-AnCora${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

CORPUS_DIR="data/corpus/UD_Spanish-AnCora"
CORPUS_FILE="$CORPUS_DIR/es_ancora-ud-train.conllu"
TEMP_DIR="data/corpus/.temp_ud_ancora"

# Verificar si ya existe
if [ -f "$CORPUS_FILE" ]; then
    SIZE=$(du -h "$CORPUS_FILE" | cut -f1)
    echo -e "${GREEN}✓ Corpus ya descargado ($SIZE)${NC}"
    echo -e "  Ubicación: $CORPUS_FILE"
    echo ""
    read -p "¿Deseas volver a descargarlo? (s/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo -e "${GREEN}✓ Usando corpus existente${NC}"
        echo ""
        echo -e "Siguiente paso:"
        echo -e "  ${YELLOW}python scripts/train_final.py${NC}"
        exit 0
    fi
    rm -f "$CORPUS_FILE"
fi

# Crear directorios
mkdir -p "$CORPUS_DIR"
mkdir -p "$TEMP_DIR"

echo -e "${BLUE}📥 Descargando UD Spanish-AnCora desde GitHub...${NC}"
echo ""

# Clonar el repositorio (solo lo necesario)
echo "   Clonando repositorio UniversalDependencies/UD_Spanish-AnCora..."
if git clone --depth=1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git "$TEMP_DIR" >/dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Repositorio clonado${NC}"
else
    echo -e "   ${RED}✗ Error clonando repositorio${NC}"
    echo -e "   Verifica tu conexión a internet"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copiar archivos .conllu al directorio de destino
echo ""
echo "   Copiando archivos de corpus..."
CONLLU_FILES=$(find "$TEMP_DIR" -name "*.conllu" 2>/dev/null)

if [ -z "$CONLLU_FILES" ]; then
    echo -e "   ${RED}✗ No se encontraron archivos .conllu${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

for file in $CONLLU_FILES; do
    filename=$(basename "$file")
    cp "$file" "$CORPUS_DIR/$filename"
    SIZE=$(du -h "$CORPUS_DIR/$filename" | cut -f1)
    echo -e "   ${GREEN}✓${NC} $filename ($SIZE)"
done

# Limpiar archivos temporales
rm -rf "$TEMP_DIR"

# Verificar que el archivo principal existe
echo ""
if [ -f "$CORPUS_FILE" ]; then
    SIZE=$(du -h "$CORPUS_FILE" | cut -f1)
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}✓ Corpus descargado exitosamente${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""
    echo -e "Ubicación: $CORPUS_DIR/"
    echo -e "Tamaño del train: $SIZE"
    echo ""
    echo -e "${BLUE}Próximos pasos:${NC}"
    echo ""
    echo -e "  ${YELLOW}python scripts/train_final.py${NC}         # Entrenamiento básico"
    echo -e "  ${YELLOW}python scripts/download_literature.py${NC} # Descargar literatura"
    echo -e "  ${YELLOW}python scripts/train_with_literature.py${NC} # Entrenar con literatura"
    echo -e "  ${YELLOW}python demo.py${NC}                        # Probar generador"
    echo ""
else
    echo -e "${RED}✗ Error: No se encontró es_ancora-ud-train.conllu${NC}"
    echo -e "Archivos disponibles en $CORPUS_DIR/:"
    ls -la "$CORPUS_DIR/"
    exit 1
fi
